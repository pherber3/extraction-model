"""Scoring for typed extraction: correctness, selective prediction, calibration,
distributional metrics and paired significance tests.

Gold / prediction formats follow the shared JSONL contract (see package docs):
gold field = {"value", "kind", "span", "spans"}; prediction field = Clairic
`answers` entry ({"type", "value", "p_null", "confidence", "median", "p10",
"p90", "quantiles", ...}).
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np
from scipy import stats

from extraction.normalize import normalize_value, text_equal

_REL_TOL = 1e-6

# --------------------------------------------------------------------------
# Field correctness
# --------------------------------------------------------------------------

_UNPARSEABLE = object()


def _is_real(x: Any) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, (bool, np.bool_))


def _field_decimals(field_spec: Mapping[str, Any]) -> int | None:
    if field_spec.get("integer"):
        return 0
    d = field_spec.get("decimals")
    return None if d is None else int(d)


def _normalize(type_: str, value: Any, decimals: int | None, reference_date: str | None) -> Any:
    """Normalize one scalar; unparseable values become a sentinel that equals nothing."""
    if value is None:
        return None
    try:
        return normalize_value(type_, value, decimals=decimals, reference_date=reference_date)
    except ValueError:
        return _UNPARSEABLE


def _scalar_equal(type_: str, a: Any, b: Any, decimals: int | None) -> bool:
    if a is _UNPARSEABLE or b is _UNPARSEABLE:
        return False
    if a is None or b is None:
        return a is None and b is None
    if _is_real(a) and _is_real(b):
        a, b = float(a), float(b)
        if decimals is not None:
            return round(a, decimals) == round(b, decimals) or math.isclose(a, b, rel_tol=1e-12, abs_tol=0.0)
        return math.isclose(a, b, rel_tol=_REL_TOL, abs_tol=1e-12)
    if type_ == "text" and isinstance(a, str) and isinstance(b, str):
        return bool(text_equal(a, b))
    return a == b


def _normalize_list(items_type: str, value: Any, decimals: int | None, reference_date: str | None) -> list[Any] | None:
    """Normalize a list answer. Empty list and null are the same answer (nothing found)."""
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        value = [value]
    out = [_normalize(items_type, v, decimals, reference_date) for v in value if v is not None]
    return out or None


def field_correct(
    field_spec: Mapping[str, Any],
    gold: Mapping[str, Any],
    pred: Mapping[str, Any] | None,
    reference_date: str | None = None,
) -> bool:
    """Exact match after normalization, with null treated as an answer.

    gold null & pred null -> correct; exactly one null -> wrong. Lists are compared as
    sets (order- and duplicate-insensitive) under the item type's equality. A missing
    prediction (`pred is None`) is always wrong.
    """
    if pred is None:
        return False
    type_ = field_spec["type"]
    decimals = _field_decimals(field_spec)
    if type_ == "list":
        items_type = field_spec.get("items", "text")
        g = _normalize_list(items_type, gold.get("value"), decimals, reference_date)
        p = _normalize_list(items_type, pred.get("value"), decimals, reference_date)
        if g is None or p is None:
            return g is None and p is None
        eq = lambda x, y: _scalar_equal(items_type, x, y, decimals)  # noqa: E731
        return all(any(eq(x, y) for y in p) for x in g) and all(any(eq(y, x) for x in g) for y in p)
    g = _normalize(type_, gold.get("value"), decimals, reference_date)
    p = _normalize(type_, pred.get("value"), decimals, reference_date)
    return _scalar_equal(type_, g, p, decimals)


def answer_prob_correct(pred: Mapping[str, Any] | None) -> float | None:
    """Model's probability that its returned answer is correct.

    null answer -> p_null; otherwise (1 - p_null) * confidence when both are present,
    confidence alone, or 1 - p_null alone; None when nothing is available.
    """
    if pred is None:
        return None
    p_null = pred.get("p_null")
    conf = pred.get("confidence")
    value = pred.get("value")
    if value is None or (isinstance(value, (list, tuple)) and len(value) == 0):
        return None if p_null is None else float(p_null)
    if p_null is not None and conf is not None:
        return (1.0 - float(p_null)) * float(conf)
    if conf is not None:
        return float(conf)
    if p_null is not None:
        return 1.0 - float(p_null)
    return None


# --------------------------------------------------------------------------
# Selective prediction (risk-coverage)
# --------------------------------------------------------------------------


def _order_by_score(score: np.ndarray) -> np.ndarray:
    """Indices by descending score; stable (ties keep input order); NaN scores last."""
    s = np.asarray(score, dtype=float)
    key = np.where(np.isnan(s), np.inf, -s)
    return np.argsort(key, kind="stable")


def risk_coverage(correct: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coverage k/n and accuracy of the top-k answers by descending score, k = 1..n."""
    c = np.asarray(correct, dtype=float)
    if c.shape != np.shape(score):
        raise ValueError("correct and score must have the same shape")
    n = c.size
    if n == 0:
        return np.empty(0), np.empty(0)
    sorted_c = c[_order_by_score(score)]
    k = np.arange(1, n + 1)
    return k / n, np.cumsum(sorted_c) / k


def accuracy_at_coverage(correct: np.ndarray, score: np.ndarray, coverage: float) -> float:
    """Accuracy of the ceil(coverage * n) highest-scoring answers (at least one)."""
    if not 0.0 < coverage <= 1.0:
        raise ValueError("coverage must be in (0, 1]")
    c = np.asarray(correct, dtype=float)
    if c.size == 0:
        raise ValueError("empty input")
    k = max(1, math.ceil(coverage * c.size - 1e-9))
    return float(c[_order_by_score(score)][:k].mean())


def matched_coverage_comparison(
    correct_a: np.ndarray,
    score_a: np.ndarray,
    correct_b: np.ndarray,
    score_b: np.ndarray,
    coverages: Sequence[float] = (0.5, 0.7, 0.8, 0.9, 1.0),
) -> list[dict[str, float]]:
    """Accuracy of systems A and B at the same coverage levels."""
    out = []
    for cov in coverages:
        acc_a = accuracy_at_coverage(correct_a, score_a, cov)
        acc_b = accuracy_at_coverage(correct_b, score_b, cov)
        out.append({"coverage": float(cov), "acc_a": acc_a, "acc_b": acc_b, "diff": acc_a - acc_b})
    return out


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------


def brier(p: np.ndarray, y: np.ndarray) -> float:
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    return float(np.mean((p - y) ** 2))


def log_loss(p: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> float:
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error with equal-width bins on [0, 1] (p = 1 goes to the last bin)."""
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    if p.size == 0:
        raise ValueError("empty input")
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    counts = np.bincount(bins, minlength=n_bins)
    sum_p = np.bincount(bins, weights=p, minlength=n_bins)
    sum_y = np.bincount(bins, weights=y, minlength=n_bins)
    nz = counts > 0
    return float(np.sum(np.abs(sum_y[nz] - sum_p[nz])) / p.size)


# --------------------------------------------------------------------------
# Distributional metrics (vectorized; scalars in -> float out)
# --------------------------------------------------------------------------


def _out(x: np.ndarray) -> float | np.ndarray:
    return float(x) if np.ndim(x) == 0 else x


def pinball(level: float | np.ndarray, q: float | np.ndarray, y: float | np.ndarray) -> float | np.ndarray:
    """Quantile (pinball) loss: level*(y-q) if y >= q else (1-level)*(q-y)."""
    level, q, y = np.asarray(level, dtype=float), np.asarray(q, dtype=float), np.asarray(y, dtype=float)
    d = y - q
    return _out(np.where(d >= 0, level * d, (level - 1.0) * d))


def default_levels(n: int = 99) -> np.ndarray:
    """Equally spaced quantile levels i/(n+1), i = 1..n (0.01..0.99 for n = 99)."""
    return np.arange(1, n + 1) / (n + 1)


def crps_from_quantiles(levels: np.ndarray, q: np.ndarray, y: float | np.ndarray) -> float | np.ndarray:
    """Discretized CRPS: mean over levels of 2 * pinball. q has shape (..., L), y shape (...)."""
    levels = np.asarray(levels, dtype=float)
    q = np.asarray(q, dtype=float)
    y = np.asarray(y, dtype=float)
    if q.shape[-1] != levels.size:
        raise ValueError("last axis of q must match levels")
    loss = pinball(levels, q, y[..., None])
    return _out(np.mean(2.0 * np.asarray(loss), axis=-1))


def interval_score(lo: Any, hi: Any, y: Any, alpha: float) -> float | np.ndarray:
    """Gneiting-Raftery interval score for a central (1 - alpha) interval [lo, hi]."""
    lo, hi, y = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float), np.asarray(y, dtype=float)
    s = (hi - lo) + (2.0 / alpha) * (lo - y) * (y < lo) + (2.0 / alpha) * (y - hi) * (y > hi)
    return _out(s)


def abs_error(median: Any, y: Any) -> float | np.ndarray:
    return _out(np.abs(np.asarray(median, dtype=float) - np.asarray(y, dtype=float)))


def clopper_pearson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    a = 1.0 - conf
    lo = 0.0 if k == 0 else float(stats.beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


def coverage(lo: Any, hi: Any, y: Any, conf: float = 0.95) -> dict[str, float]:
    """Empirical interval coverage with a Clopper-Pearson `conf` CI."""
    lo, hi, y = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float), np.asarray(y, dtype=float)
    inside = (y >= lo) & (y <= hi)
    n, k = int(inside.size), int(inside.sum())
    ci_lo, ci_hi = clopper_pearson(k, n, conf)
    return {"coverage": k / n if n else float("nan"), "k": k, "n": n, "ci_lo": ci_lo, "ci_hi": ci_hi}


# --------------------------------------------------------------------------
# Paired tests
# --------------------------------------------------------------------------


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> dict[str, float]:
    """Exact McNemar test on paired correctness.

    b = A right & B wrong, c = A wrong & B right. Under H0 b ~ Binom(b + c, 1/2).
    p_two_sided = min(1, 2 * P(X <= min(b, c))); p_a_better = P(X >= b).
    """
    a = np.asarray(correct_a, dtype=bool)
    bb = np.asarray(correct_b, dtype=bool)
    if a.shape != bb.shape:
        raise ValueError("paired arrays must have the same shape")
    b = int(np.sum(a & ~bb))
    c = int(np.sum(~a & bb))
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "n_discordant": 0, "p_two_sided": 1.0, "p_a_better": 1.0}
    p_two = min(1.0, 2.0 * float(stats.binom.cdf(min(b, c), n, 0.5)))
    p_one = float(stats.binom.sf(b - 1, n, 0.5))
    return {"b": b, "c": c, "n_discordant": n, "p_two_sided": p_two, "p_a_better": p_one}


def cluster_bootstrap_diff(
    values_a: np.ndarray,
    values_b: np.ndarray,
    clusters: Sequence[Any] | np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
    stat: Callable[[np.ndarray], float] = np.mean,
) -> tuple[float, float, float]:
    """stat(A) - stat(B) with a percentile 95% CI from resampling clusters with replacement.

    Rows are paired: a resampled cluster contributes the same rows to A and B.
    """
    va, vb = np.asarray(values_a, dtype=float), np.asarray(values_b, dtype=float)
    if va.shape != vb.shape or va.shape[0] != len(clusters):
        raise ValueError("values_a, values_b, clusters must align")
    diff = float(stat(va) - stat(vb))
    _, inv = np.unique(np.asarray(clusters), return_inverse=True)
    inv = inv.ravel()
    g = int(inv.max()) + 1
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, g, size=(n_boot, g))
    if stat is np.mean:
        # Cluster-weighted means: weight of each cluster = how often it was drawn.
        w = np.stack([np.bincount(d, minlength=g) for d in draws]).astype(float)
        n_c = np.bincount(inv, minlength=g).astype(float)
        sum_d = np.bincount(inv, weights=va - vb, minlength=g)
        boot = (w @ sum_d) / (w @ n_c)
    else:
        members = [np.flatnonzero(inv == j) for j in range(g)]
        boot = np.empty(n_boot)
        for i, d in enumerate(draws):
            idx = np.concatenate([members[j] for j in d])
            boot[i] = stat(va[idx]) - stat(vb[idx])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return diff, float(lo), float(hi)


# --------------------------------------------------------------------------
# Per-field scoring over a dataset
# --------------------------------------------------------------------------


def _quantile_array(pred: Mapping[str, Any]) -> np.ndarray | None:
    q = pred.get("quantiles")
    if q is None:
        return None
    arr = np.asarray(q, dtype=float)
    return arr if arr.ndim == 1 and arr.size > 0 else None


def _distributional_metrics(pred: Mapping[str, Any], y: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    q = _quantile_array(pred)
    median = pred.get("median")
    if median is None and q is not None and q.size % 2 == 1:
        median = float(q[q.size // 2])
    if median is None and _is_real(pred.get("value")) and (q is not None or pred.get("p10") is not None):
        median = pred.get("value")
    if median is not None:
        out["abs_error"] = abs_error(median, y)
    if q is not None:
        out["crps"] = crps_from_quantiles(default_levels(q.size), q, y)
    p10, p90 = pred.get("p10"), pred.get("p90")
    if p10 is not None and p90 is not None:
        out["interval_score_80"] = interval_score(p10, p90, y, alpha=0.2)
        out["covered_80"] = bool(float(p10) <= y <= float(p90))
        out["p10"], out["p90"] = float(p10), float(p90)
    return out


def score_predictions(
    examples: Iterable[Mapping[str, Any]],
    preds_by_id: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """One row per (example id, field). Missing predictions are scored wrong (pred_missing=True)."""
    rows: list[dict[str, Any]] = []
    for ex in examples:
        req = ex["request"]
        ref = req.get("reference_date")
        preds = preds_by_id.get(ex["id"]) or {}
        for name, spec in req["fields"].items():
            gold = ex["gold"][name]
            pred = preds.get(name)
            gold_value = gold.get("value")
            row: dict[str, Any] = {
                "id": ex["id"],
                "doc_type": ex.get("doc_type"),
                "template_id": ex.get("template_id"),
                "field": name,
                "type": spec["type"],
                "gold_kind": gold.get("kind"),
                "correct": field_correct(spec, gold, pred, ref),
                "p_correct": answer_prob_correct(pred),
                "pred_value": None if pred is None else pred.get("value"),
                "gold_value": gold_value,
                "gold_is_null": gold_value is None,
                "pred_missing": pred is None,
                "p_null": None if pred is None or pred.get("p_null") is None else float(pred["p_null"]),
            }
            if pred is not None and _is_real(gold_value):
                row.update(_distributional_metrics(pred, float(gold_value)))
            rows.append(row)
    return rows


def _accuracy_by(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, dict[str, float]]:
    groups: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        groups[str(r[key])].append(bool(r["correct"]))
    return {k: {"n": len(v), "accuracy": float(np.mean(v))} for k, v in sorted(groups.items())}


def summarize(rows: Sequence[Mapping[str, Any]], n_bins: int = 15) -> dict[str, Any]:
    """Aggregate rows from `score_predictions`."""
    if not rows:
        raise ValueError("no rows to summarize")
    correct = np.array([bool(r["correct"]) for r in rows])
    out: dict[str, Any] = {
        "n": len(rows),
        "accuracy": float(correct.mean()),
        "n_pred_missing": int(sum(bool(r["pred_missing"]) for r in rows)),
        "by_type": _accuracy_by(rows, "type"),
        "by_gold_kind": _accuracy_by(rows, "gold_kind"),
        "by_doc_type": _accuracy_by(rows, "doc_type"),
    }

    nr = [r for r in rows if r["p_null"] is not None]
    if nr:
        p = np.array([r["p_null"] for r in nr])
        y = np.array([r["gold_is_null"] for r in nr], dtype=float)
        out["null_detection"] = {
            "n": len(nr),
            "brier": brier(p, y),
            "log_loss": log_loss(p, y),
            "accuracy@0.5": float(np.mean((p >= 0.5) == (y == 1.0))),
        }

    cr = [r for r in rows if r["p_correct"] is not None]
    if cr:
        p = np.array([r["p_correct"] for r in cr])
        y = np.array([r["correct"] for r in cr], dtype=float)
        out["calibration"] = {"n": len(cr), "ece": ece(p, y, n_bins), "brier": brier(p, y), "log_loss": log_loss(p, y)}

    dist: dict[str, Any] = {}
    for metric in ("abs_error", "crps", "interval_score_80"):
        vals = [r[metric] for r in rows if metric in r]
        if vals:
            dist[metric] = {"n": len(vals), "mean": float(np.mean(vals))}
    iv = [r for r in rows if "covered_80" in r]
    if iv:
        dist["coverage_80"] = coverage([r["p10"] for r in iv], [r["p90"] for r in iv], [float(r["gold_value"]) for r in iv])
    if dist:
        out["distributional"] = dist
    return out
