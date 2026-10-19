"""Stage 0 end to end: build data, train one probe per layer, score every layer on every eval suite.

Used by ``scripts/run_stage0.py`` (Colab CLI / terminal) and ``notebooks/stage0_probe_colab.ipynb``.
Outputs in ``out_dir``: ``layer_table.json`` (one row per suite x layer), ``rows_layer<best>.json``
(per-field rows at the best layer), ``summary_layer<best>.json``, ``heads/`` (probe weights, train log).
"""

from __future__ import annotations

import collections
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import probe, public_data, scoring, synth
from .encode import SPAN_TYPES


@dataclass
class Stage0Config:
    model_id: str = "Qwen/Qwen3.5-4B"
    dry_run: bool = False  # tiny random backbone on CPU, small data
    n_synth: int = 3000  # synthetic docs; one template per doc type held out for eval
    n_squad: int = 3000  # SQuAD 2.0 train paragraphs
    n_gretel: int = 2000  # Gretel synthetic finance train docs
    n_eval: int = 400  # docs per eval suite
    epochs: int = 2
    layers: list[int] | None = None  # 1-based; None = all
    out_dir: str = "results/stage0"

    def __post_init__(self) -> None:
        if self.dry_run:
            self.n_synth, self.n_squad, self.n_gretel, self.n_eval, self.epochs = 120, 60, 40, 30, 1


def describe(examples: list[dict[str, Any]]) -> str:
    kinds = collections.Counter(g["kind"] for ex in examples for g in ex["gold"].values())
    return f"{len(examples)} docs, {sum(kinds.values())} fields {dict(kinds)}"


def build_data(config: Stage0Config) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    synth_all = synth.generate(config.n_synth, seed=0)
    by_type: dict[str, list[str]] = collections.defaultdict(list)
    for template in sorted({ex["template_id"] for ex in synth_all}):
        by_type[template.split("/")[0]].append(template)
    holdout = {random.Random(0).choice(templates) for templates in by_type.values()}
    synth_train, synth_eval = synth.split_by_template(synth_all, holdout)
    train = synth_train + public_data.squad_v2("train", config.n_squad) + public_data.gretel_finance("train", config.n_gretel)
    suites = {
        "synth_unseen_templates": synth_eval[: config.n_eval],
        "squad_v2_dev": public_data.squad_v2("validation", config.n_eval, seed=1),
        "gretel_test": public_data.gretel_finance("test", config.n_eval, seed=1),
        "fda": public_data.fda(config.n_eval),
    }
    print("holdout templates:", sorted(holdout))
    print("train:", describe(train))
    for name, examples in suites.items():
        print(f"{name}:", describe(examples))
    return train, suites


def span_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rows stage 0 answers: single-span types, gold extract or null (compute fields are out of scope)."""
    return [r for r in rows if r["type"] in SPAN_TYPES and r["gold_kind"] != "compute"]


def layer_row(suite: str, layer: int, attention: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct = np.array([r["correct"] for r in rows], dtype=float)
    is_null = np.array([r["gold_kind"] == "null" for r in rows])
    p_null = np.array([np.nan if r["p_null"] is None else r["p_null"] for r in rows], dtype=float)
    answered = ~np.isnan(p_null)  # unanswered = state truncated away; counted wrong in acc
    return {
        "suite": suite,
        "layer": layer,
        "attn": attention,
        "n": len(rows),
        "acc": float(correct.mean()),
        "extract_acc": float(correct[~is_null].mean()) if (~is_null).any() else float("nan"),
        "null_acc": float(correct[is_null].mean()) if is_null.any() else float("nan"),
        "null_brier": scoring.brier(p_null[answered], is_null[answered].astype(float)),
    }


def print_table(table: list[dict[str, Any]], suites: list[str]) -> None:
    for suite in suites:
        print(f"\n== {suite}")
        print(f"{'layer':>5} {'attn':>6} {'n':>5} {'acc':>6} {'extract':>8} {'null':>6} {'brier':>6}")
        for r in (r for r in table if r["suite"] == suite):
            print(
                f"{r['layer']:>5} {r['attn']:>6} {r['n']:>5} {r['acc']:6.3f} {r['extract_acc']:8.3f} "
                f"{r['null_acc']:6.3f} {r['null_brier']:6.3f}"
            )


def run(config: Stage0Config) -> dict[str, Any]:
    out = Path(config.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(config), indent=1))
    train, suites = build_data(config)

    if config.dry_run:
        backbone = probe.tiny_backbone(config.model_id)
    else:
        backbone = probe.load_backbone(config.model_id, device="cuda", dtype=torch.bfloat16)
        print(f"GPU memory after load: {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
    layers = config.layers or list(range(1, backbone.num_layers + 1))
    full_attention = {i + 1 for i, t in enumerate(backbone.layer_types) if t == "full_attention"}
    print(f"{backbone.num_layers} layers; full attention after layers {sorted(full_attention)}", flush=True)
    probe_config = probe.ProbeConfig(layers=layers, epochs=config.epochs, max_tokens=4096 if config.dry_run else 16384)

    started = time.perf_counter()
    result = probe.train_probes(backbone, train, probe_config)
    print(f"training: {time.perf_counter() - started:.0f}s", flush=True)
    probe.save_heads(result, out / "heads")

    table, rows_by = [], {}
    for suite, examples in suites.items():
        started = time.perf_counter()
        predictions = probe.predict(backbone, result.heads, examples, probe_config)
        for layer in layers:
            rows = span_rows(scoring.score_predictions(examples, predictions[layer]))
            rows_by[(suite, layer)] = rows
            table.append(layer_row(suite, layer, "full" if layer in full_attention else "linear", rows))
        print(f"eval {suite}: {time.perf_counter() - started:.0f}s", flush=True)
    (out / "layer_table.json").write_text(json.dumps(table, indent=1))
    print_table(table, list(suites))

    best = max(layers, key=lambda layer: np.mean([r["acc"] for r in table if r["layer"] == layer]))
    print(f"\nbest layer by mean accuracy across suites: {best} ({'full' if best in full_attention else 'linear'} attention)")
    summaries = {suite: scoring.summarize(rows_by[(suite, best)]) for suite in suites}
    (out / f"summary_layer{best}.json").write_text(json.dumps(summaries, indent=1, default=float))
    (out / f"rows_layer{best}.json").write_text(json.dumps({s: rows_by[(s, best)] for s in suites}, default=str))
    for suite in suites:
        print(f"\n== {suite} @ layer {best}: acc {summaries[suite]['accuracy']:.3f}")
        for r in [r for r in rows_by[(suite, best)] if not r["correct"]][:5]:
            p_null = "n/a" if r["p_null"] is None else f"{r['p_null']:.2f}"
            print(f"  {r['id']} {r['field']}: gold={r['gold_value']!r} pred={r['pred_value']!r} p_null={p_null}")
    return {"best_layer": best, "table": table}
