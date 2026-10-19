"""Span and null heads that read one backbone layer.

For field ``f`` with query vector ``q_f`` and state-token states ``h_i`` (after LayerNorm):

    start_logit(i) = <W_s q_f, K_s h_i> / sqrt(d)
    end_logit(j)   = <W_e q_f, K_e h_j> / sqrt(d)
    null_logit     = MLP(q_f)

``q_f`` combines the mean over the field's instruction tokens, the field block's last token and
the sequence's last token. The state states never see the schema (the state precedes it), so all
conditioning on the field enters through ``q_f``. The span is decoded as the best (s, e) with
s <= e < s + max_span_tokens.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import nn

from .encode import EncodedRecord


@dataclass
class FieldLogits:
    start: torch.Tensor  # [n_state]
    end: torch.Tensor  # [n_state]
    null: torch.Tensor  # scalar


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

    def forward(self, hidden: torch.Tensor, records: list[EncodedRecord]) -> list[list[FieldLogits]]:
        """``hidden``: [batch, seq, hidden] for this layer, right-padded like the records."""
        hidden = self.norm(hidden.float())
        scale = 1.0 / math.sqrt(self.width)
        output = []
        for row, record in enumerate(records):
            states = hidden[row, record.state_range[0] : record.state_range[1]]
            global_vector = hidden[row, len(record.input_ids) - 1]
            queries = self.query(
                torch.stack(
                    [
                        torch.cat(
                            [
                                hidden[row, field.instruction_span[0] : field.instruction_span[1]].mean(dim=0),
                                hidden[row, field.end_index],
                                global_vector,
                            ]
                        )
                        for field in record.fields
                    ]
                )
            )
            start_logits = self.start_query(queries) @ self.start_key(states).T * scale
            end_logits = self.end_query(queries) @ self.end_key(states).T * scale
            null_logits = self.null(queries).squeeze(-1)
            output.append(
                [FieldLogits(start_logits[i], end_logits[i], null_logits[i]) for i in range(len(record.fields))]
            )
        return output


def field_loss(logits: FieldLogits, kind: str | None, start_token: int, end_token: int) -> torch.Tensor | None:
    """Null BCE for every labelled field; start/end CE only for single-span extract targets.

    Compute fields carry a value that is not a span, so they train the null head (not null) only.
    """
    if kind is None:
        return None
    loss = functional.binary_cross_entropy_with_logits(
        logits.null, torch.tensor(1.0 if kind == "null" else 0.0, device=logits.null.device)
    )
    if start_token >= 0:
        loss = loss + functional.cross_entropy(logits.start.unsqueeze(0), torch.tensor([start_token], device=logits.start.device))
        loss = loss + functional.cross_entropy(logits.end.unsqueeze(0), torch.tensor([end_token], device=logits.end.device))
    return loss


def decode_span(logits: FieldLogits, max_span_tokens: int = 48) -> tuple[int, int, float]:
    """Best (start, end) with start <= end < start + max_span_tokens and its probability p(s)p(e)."""
    start_logp = functional.log_softmax(logits.start.float(), dim=-1)
    end_logp = functional.log_softmax(logits.end.float(), dim=-1)
    n = start_logp.shape[0]
    padded_end = functional.pad(end_logp, (0, max_span_tokens - 1), value=float("-inf"))
    pair = start_logp[:, None] + padded_end.unfold(0, max_span_tokens, 1)  # [n, L]: end = start + k
    flat = int(torch.argmax(pair))
    start, length = divmod(flat, max_span_tokens)
    return start, start + length, float(pair[start, length].exp())
