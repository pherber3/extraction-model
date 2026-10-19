"""Training and prediction for heads on a frozen backbone (stages 0 and 1).

One backbone forward per batch feeds every head being trained, so several heads (e.g. one probe per
layer in stage 0) cost one backbone pass. A head exposes ``layers`` (the hidden states it reads),
``loss(states, fields)`` and ``decode(states, fields, max_span_tokens)``; see ``extraction.heads``.
Predictions come out in Clairic's answer format so the same scorer handles our model and Clairic.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Protocol

import torch

from .encode import SPAN_TYPES, EncodedRecord, encode_example, token_span_to_chars
from .heads import FieldBatch, Prediction
from .normalize import normalize_value


class Head(Protocol):
    layers: tuple[int, ...]

    def loss(self, states: dict[int, torch.Tensor], fields: FieldBatch) -> torch.Tensor | None: ...

    def decode(self, states: dict[int, torch.Tensor], fields: FieldBatch, max_span_tokens: int) -> Prediction: ...


@dataclass
class Backbone:
    model: torch.nn.Module  # the text model (no LM head); forward returns hidden_states
    tokenizer: Any
    hidden_size: int
    layer_types: list[str]
    device: torch.device

    @property
    def num_layers(self) -> int:
        return len(self.layer_types)


def load_backbone(model_id: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16) -> Backbone:
    """Load a Qwen3.5 checkpoint and keep only its text model, frozen."""
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    torch.set_float32_matmul_precision("high")  # TF32 for the fp32 heads (A100 tensor cores)
    full = Qwen3_5ForConditionalGeneration.from_pretrained(model_id, dtype=dtype, device_map={"": device})
    text_model = full.model.language_model
    full.model.visual = None  # text-only for now; free the vision tower
    text_model.requires_grad_(False).eval()
    config = text_model.config
    return Backbone(text_model, AutoTokenizer.from_pretrained(model_id), config.hidden_size, list(config.layer_types), torch.device(device))


def tiny_backbone(tokenizer_id: str = "Qwen/Qwen3.5-4B", seed: int = 0) -> Backbone:
    """Randomly initialised 4-layer Qwen3.5 text model with the real tokenizer, for CPU smoke runs."""
    from transformers import AutoTokenizer, Qwen3_5TextConfig, Qwen3_5TextModel

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    torch.manual_seed(seed)
    config = Qwen3_5TextConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, linear_key_head_dim=16, linear_value_head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
        vocab_size=len(tokenizer), layer_types=["linear_attention"] * 3 + ["full_attention"],
    )
    model = Qwen3_5TextModel(config).requires_grad_(False).eval()
    return Backbone(model, tokenizer, config.hidden_size, list(config.layer_types), torch.device("cpu"))


def truncate_backbone(backbone: Backbone, deepest_layer: int) -> None:
    """Drop the decoder layers after ``deepest_layer + 1`` to save compute.

    One extra layer is kept because the model's last hidden state is the final-normed output; with
    ``deepest_layer + 1`` layers, ``hidden_states[1..deepest_layer]`` are the same raw residual states as
    in the full model (causal model, no later layer feeds back).
    """
    keep = min(deepest_layer + 1, backbone.num_layers)
    backbone.model.layers = torch.nn.ModuleList(list(backbone.model.layers)[:keep])
    backbone.model.config.num_hidden_layers = keep
    backbone.model.config.layer_types = backbone.layer_types[:keep]
    backbone.layer_types = backbone.layer_types[:keep]


def shuffled_fields(example: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Same example with the schema's field order permuted, so no field learns to rely on its position."""
    names = list(example["request"]["fields"])
    rng.shuffle(names)
    request = dict(example["request"], fields={name: example["request"]["fields"][name] for name in names})
    return dict(example, request=request)


def token_batches(records: list[EncodedRecord], max_tokens: int) -> Iterator[list[EncodedRecord]]:
    """Length-sorted batches whose padded size (rows x longest) stays under ``max_tokens``."""
    ordered = sorted(records, key=lambda record: len(record.input_ids))
    batch: list[EncodedRecord] = []
    for record in ordered:
        longest = max([len(record.input_ids)] + [len(r.input_ids) for r in batch])
        if batch and longest * (len(batch) + 1) > max_tokens:
            yield batch
            batch = []
        batch.append(record)
    if batch:
        yield batch


def padded_length(records: list[EncodedRecord], multiple: int = 128) -> int:
    """Longest record rounded up, so kernels that specialize on sequence length see few distinct shapes."""
    longest = max(len(record.input_ids) for record in records)
    return -(-longest // multiple) * multiple


@torch.no_grad()
def layer_states(
    backbone: Backbone, records: list[EncodedRecord], layers: list[int], seq_len: int
) -> dict[int, torch.Tensor]:
    """Hidden states after each requested layer (1-based; ``num_layers`` is the final, normed output).

    Rows are right-padded to ``seq_len``; the model is causal, so padding never affects real tokens.
    """
    pad = backbone.tokenizer.pad_token_id or 0
    input_ids = torch.full((len(records), seq_len), pad, dtype=torch.long)
    attention_mask = torch.zeros((len(records), seq_len), dtype=torch.long)
    for row, record in enumerate(records):
        input_ids[row, : len(record.input_ids)] = torch.tensor(record.input_ids)
        attention_mask[row, : len(record.input_ids)] = 1
    output = backbone.model(
        input_ids=input_ids.to(backbone.device),
        attention_mask=attention_mask.to(backbone.device),
        output_hidden_states=True,
        use_cache=False,
    )
    return {layer: output.hidden_states[layer] for layer in layers}


@dataclass
class TrainConfig:
    epochs: int = 2
    lr: float = 3e-4
    weight_decay: float = 0.01
    schedule: str = "constant"  # or "cosine": linear warmup then cosine decay to 0
    warmup_fraction: float = 0.05
    max_tokens: int = 8192
    max_length: int = 8192
    max_span_tokens: int = 48
    seed: int = 0
    log_every: int = 20


@dataclass
class TrainResult:
    heads: dict[str, torch.nn.Module]
    train_log: list[dict[str, float]] = field(default_factory=list)


def _needed_layers(heads: dict[str, Any]) -> list[int]:
    return sorted({layer for head in heads.values() for layer in head.layers})


def train_heads(
    backbone: Backbone, heads: dict[str, torch.nn.Module], examples: list[dict[str, Any]], config: TrainConfig
) -> TrainResult:
    rng = random.Random(config.seed)
    layers = _needed_layers(heads)
    parameters = [p for head in heads.values() for p in head.parameters()]
    optimizer = torch.optim.AdamW(parameters, lr=config.lr, weight_decay=config.weight_decay)
    # The step count is fixed by the data, so the schedule can be laid out before training.
    probe_records = [encode_example(backbone.tokenizer, ex, config.max_length) for ex in examples]
    total_steps = config.epochs * sum(1 for _ in token_batches(probe_records, config.max_tokens))
    warmup = max(1, int(config.warmup_fraction * total_steps))

    def lr_factor(step: int) -> float:
        if config.schedule == "constant":
            return 1.0
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    for head in heads.values():
        head.train()
    log: list[dict[str, float]] = []
    step = 0
    for epoch in range(config.epochs):
        records = [encode_example(backbone.tokenizer, shuffled_fields(ex, rng), config.max_length) for ex in examples]
        batches = list(token_batches(records, config.max_tokens))
        rng.shuffle(batches)
        for batch in batches:
            seq_len = padded_length(batch)
            states = layer_states(backbone, batch, layers, seq_len)
            fields = FieldBatch.build(batch, seq_len, backbone.device)
            losses = {name: loss for name, head in heads.items() if (loss := head.loss(states, fields)) is not None}
            if not losses:
                continue
            optimizer.zero_grad(set_to_none=True)
            torch.stack(list(losses.values())).sum().backward()
            optimizer.step()
            scheduler.step()
            step += 1
            if step % config.log_every == 0:
                values = {name: loss.item() for name, loss in losses.items()}
                log.append({"epoch": epoch, "step": step, **{f"loss_{name}": v for name, v in values.items()}})
                best = min(values, key=values.__getitem__)
                print(f"epoch {epoch} step {step}/{total_steps}: best {best} loss {values[best]:.4f}", flush=True)
    return TrainResult(heads, log)


def _typed_value(type_: str, raw: str, spec: dict[str, Any], reference_date: str | None) -> Any:
    try:
        return normalize_value(type_, raw, spec.get("decimals"), reference_date)
    except (ValueError, TypeError, KeyError):
        return raw


@torch.no_grad()
def predict(
    backbone: Backbone,
    heads: dict[str, torch.nn.Module],
    examples: list[dict[str, Any]],
    config: TrainConfig,
) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    """Per head: ``{example_id: {field: answer}}`` in Clairic's answer format.

    Only single-span field types are answered; others (list, estimate, forecast) are omitted, so
    the scorer counts them as unanswered rather than as wrong guesses.
    """
    by_id = {str(ex["id"]): ex for ex in examples}
    records = [encode_example(backbone.tokenizer, ex, config.max_length) for ex in examples]
    layers = _needed_layers(heads)
    for head in heads.values():
        head.eval()
    predictions: dict[str, dict[str, dict[str, dict[str, Any]]]] = {name: {} for name in heads}
    for batch in token_batches(records, config.max_tokens):
        seq_len = padded_length(batch)
        states = layer_states(backbone, batch, layers, seq_len)
        fields = FieldBatch.build(batch, seq_len, backbone.device)
        owners = [(record, encoded) for record in batch for encoded in record.fields]
        for name, head in heads.items():
            decoded = head.decode(states, fields, config.max_span_tokens)
            relative_start = (decoded.start - fields.state_start).tolist()
            relative_end = (decoded.end - fields.state_start).tolist()
            p_span, p_null = decoded.p_span.tolist(), decoded.p_null.tolist()
            for f, (record, encoded) in enumerate(owners):
                if encoded.type not in SPAN_TYPES or not record.state_offsets:
                    continue
                example = by_id[record.record_id]
                state = example["request"]["state"]
                spec = example["request"]["fields"][encoded.name]
                char_start, char_end = token_span_to_chars(state, record.state_offsets, relative_start[f], relative_end[f])
                guess = _typed_value(encoded.type, state[char_start:char_end], spec, example["request"].get("reference_date"))
                predictions[name].setdefault(record.record_id, {})[encoded.name] = {
                    "type": encoded.type,
                    "p_null": p_null[f],
                    "span": [char_start, char_end],
                    "guess": guess,
                    "confidence": p_span[f],
                    "value": None if p_null[f] > 0.5 else guess,
                }
    return predictions


def save_heads(result: TrainResult, directory: str | Path) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, head in result.heads.items():
        torch.save(head.state_dict(), directory / f"head_{name}.pt")
    (directory / "train_log.json").write_text(json.dumps(result.train_log))
