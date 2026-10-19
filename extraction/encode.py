"""Turn a Clairic-style request into one token sequence plus the positions the heads read.

Layout (one causal pass): system prompt, ``STATE:`` + the document, ``SCHEMA FIELDS:`` + one
block per field, then an assistant suffix. The document comes first, so every field block can
attend to all of it, and the document's hidden states do not depend on the schema.

Each field block yields two read positions: the mean over its instruction tokens and the
last token of the block. Gold character spans are mapped to state-token indices here, so the
heads only ever see token indices and the decoder maps them back to characters.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

SYSTEM_PROMPT = (
    "Read the complete state and schema. For every field, extract, compute, or estimate a value "
    "of the field's type from the state, or report that the state does not contain it."
)
SPAN_TYPES = frozenset({"text", "number", "date", "time", "datetime", "duration"})
CONSTRAINT_KEYS = ("unit", "integer", "decimals", "items", "min", "max", "step", "scale", "horizon", "interval")


@dataclass(frozen=True)
class EncodedField:
    name: str
    type: str
    instruction_span: tuple[int, int]  # absolute token range of the instruction text
    end_index: int  # absolute index of the last token of the field block
    kind: str | None  # gold kind: extract | compute | null; None when unlabelled
    start_token: int  # gold start, index into state tokens; -1 when not a single-span target
    end_token: int  # gold end (inclusive), index into state tokens; -1 when not a single-span target


@dataclass(frozen=True)
class EncodedRecord:
    record_id: str
    input_ids: tuple[int, ...]
    state_range: tuple[int, int]  # absolute token range of the (possibly truncated) state
    state_offsets: tuple[tuple[int, int], ...]  # char offsets into the state, one per state token
    fields: tuple[EncodedField, ...]
    truncated: bool


def _tokens(tokenizer: Any, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False).input_ids


def _constraint_text(spec: dict[str, Any]) -> str:
    parts = [f"{key.upper()}: {json.dumps(spec[key])}" for key in CONSTRAINT_KEYS if key in spec]
    return "".join(f"{part}\n" for part in parts)


def char_span_to_tokens(offsets: tuple[tuple[int, int], ...], start: int, end: int) -> tuple[int, int] | None:
    """Smallest inclusive token range covering chars [start, end); None if it is not fully kept."""
    first = next((i for i, (_, token_end) in enumerate(offsets) if token_end > start), None)
    if first is None:
        return None
    last = first
    for i in range(first, len(offsets)):
        if offsets[i][0] >= end:
            break
        last = i
    if offsets[last][1] < end:
        return None
    return first, last


def token_span_to_chars(state: str, offsets: tuple[tuple[int, int], ...], start: int, end: int) -> tuple[int, int]:
    """Char range covered by state tokens [start, end], with surrounding whitespace removed."""
    char_start, char_end = offsets[start][0], offsets[end][1]
    while char_start < char_end and state[char_start].isspace():
        char_start += 1
    while char_end > char_start and state[char_end - 1].isspace():
        char_end -= 1
    return char_start, char_end


def encode_example(
    tokenizer: Any,
    example: dict[str, Any],
    max_length: int = 8192,
) -> EncodedRecord:
    """Encode one example (``{"id", "request", "gold"?}``); gold is optional."""
    request = example["request"]
    gold = example.get("gold") or {}
    state = request.get("state", "")

    schema_ids = _tokens(tokenizer, "\n\nSCHEMA FIELDS:\n")
    blocks: list[tuple[str, str, int, int, int]] = []
    for index, (name, spec) in enumerate(request["fields"].items()):
        header = f"\nFIELD {index + 1}\nID: {name}\nTYPE: {spec['type']}\n{_constraint_text(spec)}INSTRUCTION: "
        schema_ids.extend(_tokens(tokenizer, header))
        instruction_start = len(schema_ids)
        schema_ids.extend(_tokens(tokenizer, spec.get("instructions") or name))
        instruction_end = len(schema_ids)
        schema_ids.extend(_tokens(tokenizer, f"\nEND FIELD {index + 1}\n"))
        blocks.append((name, spec["type"], instruction_start, instruction_end, len(schema_ids) - 1))

    reference = f"REFERENCE DATE: {request['reference_date']}\n" if request.get("reference_date") else ""
    prefix_ids = _tokens(
        tokenizer, f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{reference}STATE:\n"
    )
    suffix_ids = _tokens(tokenizer, "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nEXTRACTED FIELDS:")

    encoded_state = tokenizer(state, add_special_tokens=False, return_offsets_mapping=True)
    state_ids = list(encoded_state.input_ids)
    offsets = tuple(tuple(pair) for pair in encoded_state.offset_mapping)
    budget = max_length - len(prefix_ids) - len(schema_ids) - len(suffix_ids)
    if budget <= 0:
        raise ValueError(f"schema needs {max_length - budget} tokens before the state; max_length is {max_length}")
    truncated = len(state_ids) > budget
    state_ids, offsets = state_ids[:budget], offsets[:budget]

    state_start = len(prefix_ids)
    schema_offset = state_start + len(state_ids)
    fields = []
    for name, type_, instruction_start, instruction_end, end_index in blocks:
        label = gold.get(name)
        kind = label.get("kind") if label else None
        start_token = end_token = -1
        if label and kind == "extract" and type_ in SPAN_TYPES and label.get("span"):
            covered = char_span_to_tokens(offsets, *label["span"])
            if covered is not None:
                start_token, end_token = covered
        fields.append(
            EncodedField(
                name=name,
                type=type_,
                instruction_span=(instruction_start + schema_offset, instruction_end + schema_offset),
                end_index=end_index + schema_offset,
                kind=kind,
                start_token=start_token,
                end_token=end_token,
            )
        )
    return EncodedRecord(
        record_id=str(example.get("id", "unknown")),
        input_ids=tuple(prefix_ids + state_ids + schema_ids + suffix_ids),
        state_range=(state_start, schema_offset),
        state_offsets=offsets,
        fields=tuple(fields),
        truncated=truncated,
    )
