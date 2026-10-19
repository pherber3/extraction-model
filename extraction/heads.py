"""Heads that read a frozen backbone's hidden states, vectorized over every field in a batch.

Common setup. For field ``f`` the query input is the mean over its instruction tokens, the last token
of its field block and the sequence's last token. The state precedes the schema, so state-token states
never see the schema and all conditioning on the field enters through the query. Spans are pointers
over the field's own state tokens; the null head is an MLP on the query.

- ``SpanNullHead`` (stage 0 probe): one layer, independent start and end pointers.
- ``ExtractionHead`` (stage 1): a learned mix of several layers, a field mixer (transformer decoder
  layers: self-attention across the record's fields, cross-attention to its state tokens), and an end
  pointer conditioned on the start token (teacher-forced in training, top-k starts at decode time).

All index bookkeeping lives in ``FieldBatch``, built once per batch and shared by every head, so heads
run a fixed handful of tensor ops with no per-field Python loop and no host-device syncs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import nn

from .encode import EncodedRecord

MASKED = -1e9  # logit for excluded positions; finite so fully masked rows stay NaN-free


@dataclass
class FieldBatch:
    """Per-field index tensors for one padded batch ([B, T] tokens, F fields in record order)."""

    n_rows: int
    seq_len: int
    max_fields: int  # most fields in one record
    row: torch.Tensor  # [F] long: batch row of each field
    slot: torch.Tensor  # [F] long: position of the field within its record
    instruction_weights: torch.Tensor  # [F, B*T] float: 1/len on the field's instruction tokens
    end_index: torch.Tensor  # [F] long: flat index (row*T + pos) of the field block's last token
    last_index: torch.Tensor  # [F] long: flat index of the sequence's last token
    state_mask: torch.Tensor  # [F, T] bool: positions that are state tokens of the field's row
    row_memory_mask: torch.Tensor  # [B, T] bool: state tokens plus the last token (never empty)
    state_start: torch.Tensor  # [F] long: absolute position of the row's first state token
    labelled: torch.Tensor  # [F] float: 1 when the field has a gold kind
    n_labelled: int  # host-side count of labelled fields, so the loss needs no device sync
    is_null: torch.Tensor  # [F] float: gold kind is null
    start_target: torch.Tensor  # [F] long: absolute gold start position, -1 when none
    end_target: torch.Tensor  # [F] long: absolute gold end position (inclusive), -1 when none

    @classmethod
    def build(cls, records: list[EncodedRecord], seq_len: int, device: torch.device) -> FieldBatch:
        rows, slots, ends, lasts, starts, labelled, is_null, start_t, end_t = [], [], [], [], [], [], [], [], []
        weight_index, weight_value = [], []
        state_mask, memory_mask = [], []
        for row, record in enumerate(records):
            base = row * seq_len
            state_start, state_end = record.state_range
            mask = torch.zeros(seq_len, dtype=torch.bool)
            mask[state_start:state_end] = True
            memory = mask.clone()
            memory[len(record.input_ids) - 1] = True
            memory_mask.append(memory)
            for slot, field in enumerate(record.fields):
                f = len(rows)
                rows.append(row)
                slots.append(slot)
                first, stop = field.instruction_span
                weight_index.extend((f, base + position) for position in range(first, stop))
                weight_value.extend([1.0 / (stop - first)] * (stop - first))
                ends.append(base + field.end_index)
                lasts.append(base + len(record.input_ids) - 1)
                starts.append(state_start)
                state_mask.append(mask)
                labelled.append(field.kind is not None)
                is_null.append(1.0 if field.kind == "null" else 0.0)
                has_span = field.start_token >= 0
                start_t.append(state_start + field.start_token if has_span else -1)
                end_t.append(state_start + field.end_token if has_span else -1)
        weights = torch.zeros(len(rows), len(records) * seq_len)
        if weight_index:
            index = torch.tensor(weight_index)
            weights[index[:, 0], index[:, 1]] = torch.tensor(weight_value)

        def long(values: list[int]) -> torch.Tensor:
            return torch.tensor(values, dtype=torch.long, device=device)

        return cls(
            n_rows=len(records),
            seq_len=seq_len,
            max_fields=max(len(record.fields) for record in records),
            row=long(rows),
            slot=long(slots),
            instruction_weights=weights.to(device),
            end_index=long(ends),
            last_index=long(lasts),
            state_mask=torch.stack(state_mask).to(device),
            row_memory_mask=torch.stack(memory_mask).to(device),
            state_start=long(starts),
            labelled=torch.tensor(labelled, dtype=torch.float, device=device),
            n_labelled=sum(labelled),
            is_null=torch.tensor(is_null, device=device),
            start_target=long(start_t),
            end_target=long(end_t),
        )


@dataclass
class Prediction:
    """Per-field decoded answer, all [F]: absolute token positions, p(span) and p(null)."""

    start: torch.Tensor
    end: torch.Tensor
    p_span: torch.Tensor
    p_null: torch.Tensor


def _queries_input(flat: torch.Tensor, fields: FieldBatch) -> torch.Tensor:
    return torch.cat([fields.instruction_weights @ flat, flat[fields.end_index], flat[fields.last_index]], dim=-1)


def _row_scores(queries: torch.Tensor, keys: torch.Tensor, rows: torch.Tensor, fields: FieldBatch) -> torch.Tensor:
    """Dot products of each query with every token, keeping the query's own row: [..., T].

    ``queries`` [N, d] scores against ``keys`` [B*T, d] in one matmul ([N, B*T]); then each query keeps
    the T scores of row ``rows[n]``. Cheaper in memory than gathering per-row keys ([N, T, d]).
    """
    scores = (queries @ keys.T / math.sqrt(queries.shape[-1])).view(-1, fields.n_rows, fields.seq_len)
    return scores[torch.arange(scores.shape[0], device=scores.device), rows]


def _null_and_span_terms(
    null: torch.Tensor, start: torch.Tensor, end: torch.Tensor, fields: FieldBatch
) -> torch.Tensor | None:
    """Mean over labelled fields of: null BCE + start CE + end CE (span terms for span targets only).

    Compute fields carry a value that is not a span, so they train the null head (not null) only.
    Masks and ``ignore_index`` avoid data-dependent shapes, so the loss never syncs the device.
    """
    if fields.n_labelled == 0:
        return None
    total = (functional.binary_cross_entropy_with_logits(null, fields.is_null, reduction="none") * fields.labelled).sum()
    total = total + functional.cross_entropy(start, fields.start_target, ignore_index=-1, reduction="sum")
    total = total + functional.cross_entropy(end, fields.end_target, ignore_index=-1, reduction="sum")
    return total / fields.n_labelled


class SpanNullHead(nn.Module):
    """Stage-0 probe on a single layer: independent start and end pointers."""

    def __init__(self, layer: int, hidden_size: int, width: int = 512) -> None:
        super().__init__()
        self.layers = (layer,)
        self.width = width
        self.norm = nn.LayerNorm(hidden_size)
        self.query = nn.Linear(3 * hidden_size, width)
        self.start_query = nn.Linear(width, width, bias=False)
        self.end_query = nn.Linear(width, width, bias=False)
        self.start_key = nn.Linear(hidden_size, width, bias=False)
        self.end_key = nn.Linear(hidden_size, width, bias=False)
        self.null = nn.Sequential(nn.GELU(), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))

    def _logits(self, states: dict[int, torch.Tensor], fields: FieldBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = states[self.layers[0]]
        flat = self.norm(hidden.float()).reshape(fields.n_rows * fields.seq_len, -1)
        queries = self.query(_queries_input(flat, fields))
        start = _row_scores(self.start_query(queries), self.start_key(flat), fields.row, fields)
        end = _row_scores(self.end_query(queries), self.end_key(flat), fields.row, fields)
        return (
            start.masked_fill(~fields.state_mask, MASKED),
            end.masked_fill(~fields.state_mask, MASKED),
            self.null(queries).squeeze(-1),
        )

    def loss(self, states: dict[int, torch.Tensor], fields: FieldBatch) -> torch.Tensor | None:
        start, end, null = self._logits(states, fields)
        return _null_and_span_terms(null, start, end, fields)

    @torch.no_grad()
    def decode(self, states: dict[int, torch.Tensor], fields: FieldBatch, max_span_tokens: int = 48) -> Prediction:
        """Best (start, end) with start <= end < start + max_span_tokens, scored by p(s) p(e)."""
        start, end, null = self._logits(states, fields)
        start_logp = functional.log_softmax(start, dim=-1)
        end_logp = functional.log_softmax(end, dim=-1)
        padded_end = functional.pad(end_logp, (0, max_span_tokens - 1), value=float("-inf"))
        pair = start_logp[:, :, None] + padded_end.unfold(1, max_span_tokens, 1)  # [F, T, L]: end = start + k
        best, flat = pair.flatten(1).max(dim=-1)
        first = flat // max_span_tokens
        return Prediction(first, first + flat % max_span_tokens, best.exp(), torch.sigmoid(null))


class ExtractionHead(nn.Module):
    """Stage-1 head: layer mix + field mixer + start-conditioned end pointer."""

    def __init__(
        self,
        layers: tuple[int, ...],
        hidden_size: int,
        width: int = 768,
        mixer_layers: int = 2,
        attention_heads: int = 8,
        dropout: float = 0.1,
        top_k_starts: int = 8,
    ) -> None:
        super().__init__()
        self.layers = tuple(layers)
        self.top_k_starts = top_k_starts
        self.layer_norms = nn.ModuleList(nn.LayerNorm(hidden_size) for _ in self.layers)
        self.layer_weights = nn.Parameter(torch.zeros(len(self.layers)))
        self.query = nn.Linear(3 * hidden_size, width)
        self.memory = nn.Linear(hidden_size, width)
        self.mixer = nn.ModuleList(
            nn.TransformerDecoderLayer(
                width, attention_heads, 4 * width, dropout, activation="gelu", batch_first=True, norm_first=True
            )
            for _ in range(mixer_layers)
        )
        self.field_norm = nn.LayerNorm(width)
        self.start_query = nn.Linear(width, width, bias=False)
        self.start_key = nn.Linear(hidden_size, width, bias=False)
        self.end_query = nn.Sequential(nn.Linear(width + hidden_size, width), nn.GELU(), nn.Linear(width, width))
        self.end_key = nn.Linear(hidden_size, width, bias=False)
        self.null = nn.Sequential(nn.GELU(), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))

    def mixed(self, states: dict[int, torch.Tensor], fields: FieldBatch) -> torch.Tensor:
        """Softmax-weighted sum of per-layer-normalized states: [B*T, H] float32."""
        weights = torch.softmax(self.layer_weights, dim=0)
        total = sum(
            w * norm(states[layer].float()) for w, norm, layer in zip(weights, self.layer_norms, self.layers)
        )
        return total.reshape(fields.n_rows * fields.seq_len, -1)

    def _encode(self, states: dict[int, torch.Tensor], fields: FieldBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mixed token states [B*T, H], refined field vectors [F, width], start logits [F, T]."""
        flat = self.mixed(states, fields)
        initial = self.query(_queries_input(flat, fields))
        tokens = torch.zeros(fields.n_rows, fields.max_fields, initial.shape[-1], device=flat.device, dtype=initial.dtype)
        tokens = tokens.index_put((fields.row, fields.slot), initial)
        padding = torch.ones(fields.n_rows, fields.max_fields, dtype=torch.bool, device=flat.device)
        padding = padding.index_put((fields.row, fields.slot), torch.zeros_like(fields.row, dtype=torch.bool))
        memory = self.memory(flat).view(fields.n_rows, fields.seq_len, -1)
        for layer in self.mixer:
            tokens = layer(tokens, memory, tgt_key_padding_mask=padding, memory_key_padding_mask=~fields.row_memory_mask)
        refined = self.field_norm(tokens[fields.row, fields.slot])
        start = _row_scores(self.start_query(refined), self.start_key(flat), fields.row, fields)
        return flat, refined, start.masked_fill(~fields.state_mask, MASKED)

    def _end_logits(self, flat: torch.Tensor, refined: torch.Tensor, starts: torch.Tensor, fields: FieldBatch) -> torch.Tensor:
        """End logits given start positions. ``starts`` [F, K] absolute positions -> [F, K, T]."""
        n_fields, k = starts.shape
        start_states = flat[(fields.row[:, None] * fields.seq_len + starts).reshape(-1)]
        queries = self.end_query(torch.cat([refined.repeat_interleave(k, dim=0), start_states], dim=-1))
        rows = fields.row.repeat_interleave(k)
        end = _row_scores(queries, self.end_key(flat), rows, fields).view(n_fields, k, -1)
        positions = torch.arange(fields.seq_len, device=flat.device)
        allowed = fields.state_mask[:, None, :] & (positions[None, None, :] >= starts[:, :, None])
        return end.masked_fill(~allowed, MASKED)

    def loss(self, states: dict[int, torch.Tensor], fields: FieldBatch) -> torch.Tensor | None:
        flat, refined, start = self._encode(states, fields)
        # Teacher forcing: condition the end on the gold start (fields without a span use the argmax; their
        # end term is ignored).
        given = torch.where(fields.start_target >= 0, fields.start_target, start.argmax(dim=-1).detach())
        end = self._end_logits(flat, refined, given[:, None], fields)[:, 0]
        return _null_and_span_terms(self.null(refined).squeeze(-1), start, end, fields)

    @torch.no_grad()
    def decode(self, states: dict[int, torch.Tensor], fields: FieldBatch, max_span_tokens: int = 48) -> Prediction:
        """Top-k starts, the best end within max_span_tokens of each, scored by p(s) p(e | s)."""
        flat, refined, start = self._encode(states, fields)
        start_logp = functional.log_softmax(start, dim=-1)
        top_logp, top_start = start_logp.topk(min(self.top_k_starts, fields.seq_len), dim=-1)  # [F, K]
        end = self._end_logits(flat, refined, top_start, fields)
        positions = torch.arange(fields.seq_len, device=flat.device)
        too_long = positions[None, None, :] >= top_start[:, :, None] + max_span_tokens
        end_logp = functional.log_softmax(end, dim=-1).masked_fill(too_long, float("-inf"))
        best_end_logp, best_end = end_logp.max(dim=-1)  # [F, K]
        joint = top_logp + best_end_logp
        best, choice = joint.max(dim=-1)
        pick = choice[:, None]
        return Prediction(
            top_start.gather(1, pick)[:, 0],
            best_end.gather(1, pick)[:, 0],
            best.exp(),
            torch.sigmoid(self.null(refined).squeeze(-1)),
        )
