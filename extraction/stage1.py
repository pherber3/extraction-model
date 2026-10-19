"""Stage 1: the extraction head vs the stage-0 probe, trained side by side on one frozen backbone.

Both heads see the same batches (one backbone forward feeds both), the same data as stage 0 and the
same schedule, so the comparison isolates the head design: a learned mix of several layers, a field
mixer, and an end pointer conditioned on the start. Outputs in ``out_dir``: ``comparison.json`` (one
row per suite x head), ``summary_<head>.json``, ``rows_<head>.json``, ``layer_weights.json``, ``heads/``.
"""

from __future__ import annotations

import collections
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import scoring
from . import train as trainer
from .heads import ExtractionHead, SpanNullHead
from .stage0 import Stage0Config, build_data, load, span_rows


@dataclass
class Stage1Config(Stage0Config):
    epochs: int = 3
    lr: float = 2e-4
    max_tokens: int = 16384  # 80 GB A100; only the mixed layers' states are kept
    mix_layers: tuple[int, ...] = (16, 17, 18, 19, 20)
    probe_layer: int = 19  # stage 0's best layer
    width: int = 768
    mixer_layers: int = 2
    out_dir: str = "results/stage1"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.dry_run:  # the tiny backbone has 4 layers
            self.epochs, self.mix_layers, self.probe_layer, self.width = 1, (2, 3), 3, 64


def token_f1(gold: Any, pred: Any) -> float:
    """SQuAD-style token F1 after SQuAD normalization; 0 when the prediction is null."""
    if pred is None:
        return 0.0
    gold_tokens = scoring.squad_normalize(str(gold)).split()
    pred_tokens = scoring.squad_normalize(str(pred)).split()
    common = sum((collections.Counter(gold_tokens) & collections.Counter(pred_tokens)).values())
    if not gold_tokens or not pred_tokens or common == 0:
        return float(gold_tokens == pred_tokens)
    precision, recall = common / len(pred_tokens), common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def comparison_row(suite: str, head: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    extract = [r for r in rows if r["gold_kind"] == "extract"]
    null = [r for r in rows if r["gold_kind"] == "null"]
    summary = scoring.summarize(rows)

    def mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else float("nan")

    return {
        "suite": suite,
        "head": head,
        "n": len(rows),
        "acc": mean([r["correct"] for r in rows]),
        "acc_relaxed": mean([r["correct_relaxed"] for r in rows]),
        "extract_em": mean([r["correct"] for r in extract]),
        "extract_em_relaxed": mean([r["correct_relaxed"] for r in extract]),
        "extract_f1": mean([token_f1(r["gold_value"], r["pred_value"]) for r in extract]),
        "null_acc": mean([r["correct"] for r in null]),
        "null_brier": summary.get("null_detection", {}).get("brier", float("nan")),
        "ece": summary.get("calibration", {}).get("ece", float("nan")),
    }


def run(config: Stage1Config) -> dict[str, Any]:
    out = Path(config.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(config), indent=1))
    train, suites = build_data(config)

    backbone = load(config.model_id, config.dry_run)
    deepest = max(*config.mix_layers, config.probe_layer)
    trainer.truncate_backbone(backbone, deepest)
    print(f"backbone truncated to {backbone.num_layers} layers (deepest read: {deepest})", flush=True)
    train_config = trainer.TrainConfig(
        epochs=config.epochs,
        lr=config.lr,
        schedule="cosine",
        max_tokens=4096 if config.dry_run else config.max_tokens,
    )
    torch.manual_seed(0)
    heads = {
        f"probe_L{config.probe_layer}": SpanNullHead(config.probe_layer, backbone.hidden_size),
        "stage1": ExtractionHead(config.mix_layers, backbone.hidden_size, config.width, config.mixer_layers),
    }
    heads = {name: head.to(backbone.device) for name, head in heads.items()}
    for name, head in heads.items():
        print(f"{name}: {sum(p.numel() for p in head.parameters()) / 1e6:.1f}M parameters", flush=True)

    started = time.perf_counter()
    result = trainer.train_heads(backbone, heads, train, train_config)
    print(f"training: {time.perf_counter() - started:.0f}s", flush=True)
    trainer.save_heads(result, out / "heads")
    weights = torch.softmax(heads["stage1"].layer_weights.detach().float().cpu(), dim=0).tolist()
    (out / "layer_weights.json").write_text(json.dumps(dict(zip(map(str, config.mix_layers), weights)), indent=1))
    print("stage1 layer weights:", {layer: round(w, 3) for layer, w in zip(config.mix_layers, weights)})

    table, rows_by = [], collections.defaultdict(dict)
    for suite, examples in suites.items():
        started = time.perf_counter()
        predictions = trainer.predict(backbone, result.heads, examples, train_config)
        for name in heads:
            rows = span_rows(scoring.score_predictions(examples, predictions[name]))
            rows_by[name][suite] = rows
            table.append(comparison_row(suite, name, rows))
        print(f"eval {suite}: {time.perf_counter() - started:.0f}s", flush=True)
    (out / "comparison.json").write_text(json.dumps(table, indent=1))
    for name in heads:
        summaries = {suite: scoring.summarize(rows) for suite, rows in rows_by[name].items()}
        (out / f"summary_{name}.json").write_text(json.dumps(summaries, indent=1, default=float))
        (out / f"rows_{name}.json").write_text(json.dumps(rows_by[name], default=str))

    columns = ["acc", "acc_relaxed", "extract_em", "extract_em_relaxed", "extract_f1", "null_acc", "null_brier", "ece"]
    for suite in suites:
        print(f"\n== {suite}")
        print(f"{'head':>10} {'n':>5} " + " ".join(f"{c[:11]:>11}" for c in columns))
        for r in (r for r in table if r["suite"] == suite):
            print(f"{r['head']:>10} {r['n']:>5} " + " ".join(f"{r[c]:11.3f}" for c in columns))
    return {"table": table, "layer_weights": weights}
