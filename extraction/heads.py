"""Span and null heads that read one backbone layer, vectorized over every field in a batch.

For field ``f`` with query vector ``q_f`` and state-token states ``h_i`` (after LayerNorm):

    start_logit(i) = <W_s q_f, K_s h_i> / sqrt(d)
    end_logit(j)   = <W_e q_f, K_e h_j> / sqrt(d)
    null_logit     = MLP(q_f)

``q_f`` combines the mean over the field's instruction tokens, the field block's last token and
the sequence's last token. The state states never see the schema (the state precedes it), so all
conditioning on the field enters through ``q_f``. The span is decoded as the best (s, e) with
s <= e < s + max_span_tokens.

All index bookkeeping lives in ``FieldBatch``, built once per batch and shared by every layer's
head, so a head's forward and loss are a fixed handful of tensor ops with no per-field Python
loop and no host-device syncs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import nn

from .encode import EncodedRecord

MASKED = -1e9  # logit for non-state positions; finite so fully masked rows stay NaN-free


@dataclass
class FieldBatch:
    """Per-field index tensors for one padded batch ([B, T] tokens, F fields in record order)."""

    row: torch.Tensor  # [F] long: batch row of each field
    instruction_weights: torch.Tensor  # [F, B*T] float: 1/len on the field's instruction tokens
    end_index: torch.Tensor  # [F] long: flat index (row*T + pos) of the field block's last token
    last_index: torch.Tensor  # [F] long: flat index of the sequence's last token
    state_mask: torch.Tensor  # [F, T] bool: positions that are state tokens of the field's row
    state_start: torch.Tensor  # [F] long: absolute position of the row's first state token
    labelled: torch.Tensor  # [F] float: 1 when the field has a gold kind
    n_labelled: int  # host-side count of labelled fields, so the loss needs no device sync
    is_null: torch.Tensor  # [F] float: gold kind is null
    start_target: torch.Tensor  # [F] long: absolute gold start position, -1 when none
    end_target: torch.Tensor  # [F] long: absolute gold end position (inclusive), -1 when none

    @classmethod
    def build(cls, records: list[EncodedRecord], seq_len: int, device: torch.device) -> FieldBatch:
        rows, ends, lasts, starts, labelled, is_null, start_t, end_t = [], [], [], [], [], [], [], []
        weight_index, weight_value = [], []
        state_mask = []
        for row, record in enumerate(records):
            base = row * seq_len
            state_start, state_end = record.state_range
            mask = torch.zeros(seq_len, dtype=torch.bool)
            mask[state_start:state_end] = True
            for field in record.fields:
                f = len(rows)
                rows.append(row)
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
            row=long(rows),
            instruction_weights=weights.to(device),
            end_index=long(ends),
            last_index=long(lasts),
            state_mask=torch.stack(state_mask).to(device),
            state_start=long(starts),
            labelled=torch.tensor(labelled, dtype=torch.float, device=device),
            n_labelled=sum(labelled),
            is_null=torch.tensor(is_null, device=device),
            start_target=long(start_t),
            end_target=long(end_t),
        )


@dataclass
class HeadOutput:
    start: torch.Tensor  # [F, T] logits, MASKED outside the field's state tokens
    end: torch.Tensor  # [F, T]
    null: torch.Tensor  # [F]


class SpanNullHead(nn.Module):
    def __init__(self, hidden_size: int, width: int = 512) -> None:
        super().__init__()
        self.width = width
        self.norm = nn.LayerNorm(hidden_size)
        self.query = nn.Linear(3 * hidden_size, width)
        self.start_query = nn.Linear(width, width, bias=False)
        self.end_query = nn.Linear(width, width, bias=False)
        self.start_key = nn.Linear(hidden_size, width, bias=False)
        self.end_key = nn.Linear(hidden_size, width, bias=False)
        self.null = nn.Sequential(nn.GELU(), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))

    def forward(self, hidden: torch.Tensor, fields: FieldBatch) -> HeadOutput:
        """``hidden``: [B, T, H] for this layer, padded to the ``T`` used to build ``fields``."""
        batch, seq_len, _ = hidden.shape
        flat = self.norm(hidden.float()).reshape(batch * seq_len, -1)
        queries = self.query(
            torch.cat([fields.instruction_weights @ flat, flat[fields.end_index], flat[fields.last_index]], dim=-1)
        )
        scale = 1.0 / math.sqrt(self.width)

        def pointer(query_proj: nn.Linear, key_proj: nn.Linear) -> torch.Tensor:
            # [F, B*T] against every token in the batch, then keep each field's own row: [F, T].
            scores = (query_proj(queries) @ key_proj(flat).T * scale).view(-1, batch, seq_len)
            scores = scores[torch.arange(scores.shape[0], device=scores.device), fields.row]
            return scores.masked_fill(~fields.state_mask, MASKED)

        return HeadOutput(
            pointer(self.start_query, self.start_key),
            pointer(self.end_query, self.end_key),
            self.null(queries).squeeze(-1),
        )


def batch_loss(output: HeadOutput, fields: FieldBatch) -> torch.Tensor | None:
    """Mean over labelled fields of: null BCE + start CE + end CE (span terms for span targets only).

    Compute fields carry a value that is not a span, so they train the null head (not null) only.
    Masks and ``ignore_index`` avoid data-dependent shapes, so the loss never syncs the device.
    """
    if fields.n_labelled == 0:
        return None
    null_terms = functional.binary_cross_entropy_with_logits(output.null, fields.is_null, reduction="none")
    total = (null_terms * fields.labelled).sum()
    total = total + functional.cross_entropy(output.start, fields.start_target, ignore_index=-1, reduction="sum")
    total = total + functional.cross_entropy(output.end, fields.end_target, ignore_index=-1, reduction="sum")
    return total / fields.n_labelled


@torch.no_grad()
def decode_spans(output: HeadOutput, max_span_tokens: int = 48) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per field: best absolute (start, end) with start <= end < start + max_span_tokens, and p(s)p(e)."""
    start_logp = functional.log_softmax(output.start.float(), dim=-1)
    end_logp = functional.log_softmax(output.end.float(), dim=-1)
    padded_end = functional.pad(end_logp, (0, max_span_tokens - 1), value=float("-inf"))
    pair = start_logp[:, :, None] + padded_end.unfold(1, max_span_tokens, 1)  # [F, T, L]: end = start + k
    best, flat = pair.flatten(1).max(dim=-1)
    start = flat // max_span_tokens
    return start, start + flat % max_span_tokens, best.exp()
