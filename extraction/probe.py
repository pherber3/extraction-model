"""Stage-0 probe: which backbone layer makes span extraction and null detection readable?

The backbone is frozen. One ``SpanNullHead`` per probed layer is trained on the same batches
(one backbone forward per batch feeds every head), then each head predicts on held-out examples
in Clairic's answer format so the same scorer handles our model and Clairic.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import torch

from .encode import SPAN_TYPES, EncodedRecord, encode_example, token_span_to_chars
from .heads import FieldBatch, SpanNullHead, batch_loss, decode_spans
from .normalize import normalize_value


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

    torch.set_float32_matmul_precision("high")  # TF32 for the fp32 probe heads (A100 tensor cores)
    full = Qwen3_5ForConditionalGeneration.from_pretrained(model_id, dtype=dtype, device_map={"": device})
    text_model = full.model.language_model
    full.model.visual = None  # stage 0 is text-only; free the vision tower
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
class ProbeConfig:
    layers: list[int]
    head_width: int = 512
    epochs: int = 3
    lr: float = 3e-4
    weight_decay: float = 0.01
    max_tokens: int = 16384
    max_length: int = 8192
    max_span_tokens: int = 48
    seed: int = 0
    log_every: int = 20


@dataclass
class ProbeResult:
    heads: dict[int, SpanNullHead]
    train_log: list[dict[str, float]] = field(default_factory=list)


def train_probes(backbone: Backbone, examples: list[dict[str, Any]], config: ProbeConfig) -> ProbeResult:
    rng = random.Random(config.seed)
    torch.manual_seed(config.seed)
    heads = {layer: SpanNullHead(backbone.hidden_size, config.head_width).to(backbone.device) for layer in config.layers}
    parameters = [p for head in heads.values() for p in head.parameters()]
    optimizer = torch.optim.AdamW(parameters, lr=config.lr, weight_decay=config.weight_decay)
    log: list[dict[str, float]] = []
    step = 0
    for epoch in range(config.epochs):
        records = [encode_example(backbone.tokenizer, shuffled_fields(ex, rng), config.max_length) for ex in examples]
        batches = list(token_batches(records, config.max_tokens))
        rng.shuffle(batches)
        for batch in batches:
            seq_len = padded_length(batch)
            states = layer_states(backbone, batch, config.layers, seq_len)
            fields = FieldBatch.build(batch, seq_len, backbone.device)
            losses = {
                layer: loss
                for layer, head in heads.items()
                if (loss := batch_loss(head(states[layer], fields), fields)) is not None
            }
            if not losses:
                continue
            optimizer.zero_grad(set_to_none=True)
            torch.stack(list(losses.values())).sum().backward()
            optimizer.step()
            step += 1
            if step % config.log_every == 0:
                values = {layer: loss.item() for layer, loss in losses.items()}
                log.append({"epoch": epoch, "step": step, **{f"loss_L{layer}": v for layer, v in values.items()}})
                best = min(values, key=values.__getitem__)
                print(f"epoch {epoch} step {step}/{config.epochs * len(batches)}: best layer {best} loss {values[best]:.4f}", flush=True)
    return ProbeResult(heads, log)


def _typed_value(type_: str, raw: str, spec: dict[str, Any], reference_date: str | None) -> Any:
    try:
        return normalize_value(type_, raw, spec.get("decimals"), reference_date)
    except (ValueError, TypeError, KeyError):
        return raw


@torch.no_grad()
def predict(
    backbone: Backbone,
    heads: dict[int, SpanNullHead],
    examples: list[dict[str, Any]],
    config: ProbeConfig,
) -> dict[int, dict[str, dict[str, dict[str, Any]]]]:
    """Per layer: ``{example_id: {field: answer}}`` in Clairic's answer format.

    Only single-span field types are answered; others (list, estimate, forecast) are omitted, so
    the scorer counts them as unanswered rather than as wrong guesses.
    """
    by_id = {str(ex["id"]): ex for ex in examples}
    records = [encode_example(backbone.tokenizer, ex, config.max_length) for ex in examples]
    predictions: dict[int, dict[str, dict[str, dict[str, Any]]]] = {layer: {} for layer in heads}
    for batch in token_batches(records, config.max_tokens):
        seq_len = padded_length(batch)
        states = layer_states(backbone, batch, list(heads), seq_len)
        fields = FieldBatch.build(batch, seq_len, backbone.device)
        owners = [(record, encoded) for record in batch for encoded in record.fields]
        for layer, head in heads.items():
            output = head.eval()(states[layer], fields)
            start, end, p_span = decode_spans(output, config.max_span_tokens)
            relative_start = (start - fields.state_start).tolist()
            relative_end = (end - fields.state_start).tolist()
            p_span, p_null = p_span.tolist(), torch.sigmoid(output.null).tolist()
            for f, (record, encoded) in enumerate(owners):
                if encoded.type not in SPAN_TYPES or not record.state_offsets:
                    continue
                example = by_id[record.record_id]
                state = example["request"]["state"]
                spec = example["request"]["fields"][encoded.name]
                char_start, char_end = token_span_to_chars(state, record.state_offsets, relative_start[f], relative_end[f])
                guess = _typed_value(encoded.type, state[char_start:char_end], spec, example["request"].get("reference_date"))
                answers = predictions[layer].setdefault(record.record_id, {})
                answers[encoded.name] = {
                    "type": encoded.type,
                    "p_null": p_null[f],
                    "span": [char_start, char_end],
                    "guess": guess,
                    "confidence": p_span[f],
                    "value": None if p_null[f] > 0.5 else guess,
                }
    return predictions


def save_heads(result: ProbeResult, directory: str | Path) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for layer, head in result.heads.items():
        torch.save(head.state_dict(), directory / f"head_L{layer}.pt")
    (directory / "train_log.json").write_text(json.dumps(result.train_log))
