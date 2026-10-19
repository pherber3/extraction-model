import json

import numpy as np
import pytest
import requests

from extraction import clairic, scoring

# ---------------------------------------------------------------- correctness


@pytest.mark.parametrize(
    ("gold", "pred", "expected"),
    [
        (None, None, True),  # null is an answer
        (None, 649.0, False),  # hallucinated value
        (649.0, None, False),  # missed value
        (649.0, 649.0, True),
        (649.0, 650.0, False),
    ],
)
def test_null_as_answer_truth_table(gold, pred, expected):
    spec = {"type": "number", "decimals": 2}
    assert scoring.field_correct(spec, {"value": gold}, {"value": pred}) is expected


def test_missing_prediction_is_wrong_even_for_null_gold():
    assert scoring.field_correct({"type": "text"}, {"value": None}, None) is False


def test_number_decimals_and_default_tolerance():
    spec2 = {"type": "number", "decimals": 2}
    assert scoring.field_correct(spec2, {"value": 649.0}, {"value": 649.001})
    assert not scoring.field_correct(spec2, {"value": 649.0}, {"value": 649.01})
    assert scoring.field_correct(spec2, {"value": 649.0}, {"value": "$649.00"})
    assert not scoring.field_correct(spec2, {"value": 649.0}, {"value": "not a number"})
    plain = {"type": "number"}
    assert scoring.field_correct(plain, {"value": 1_000_000.0}, {"value": 1_000_000.5})  # rel 5e-7
    assert not scoring.field_correct(plain, {"value": 1_000_000.0}, {"value": 1_000_002.0})


def test_list_is_order_insensitive_set_match():
    spec = {"type": "list", "items": "date"}
    gold = {"value": ["2026-09-28", "2026-10-01"]}
    assert scoring.field_correct(spec, gold, {"value": ["2026-10-01", "2026-09-28"]})
    assert not scoring.field_correct(spec, gold, {"value": ["2026-10-01"]})
    assert not scoring.field_correct(spec, gold, {"value": ["2026-10-01", "2026-09-28", "2026-09-29"]})
    # empty list and null are the same answer
    assert scoring.field_correct(spec, {"value": None}, {"value": []})


def test_answer_prob_correct():
    assert scoring.answer_prob_correct({"value": None, "p_null": 0.8, "confidence": 0.3}) == 0.8
    assert scoring.answer_prob_correct({"value": 3.0, "p_null": 0.2, "confidence": 0.5}) == pytest.approx(0.4)
    assert scoring.answer_prob_correct({"value": 3.0, "confidence": 0.5}) == 0.5
    assert scoring.answer_prob_correct({"value": 3.0}) is None


# ------------------------------------------------------------ risk-coverage


def test_risk_coverage_curve_and_full_coverage_accuracy():
    correct = np.array([1, 0, 1, 0, 1], dtype=bool)
    score = np.array([0.9, 0.8, 0.7, 0.1, 0.95])
    cov, acc = scoring.risk_coverage(correct, score)
    # order by score: idx 4(1), 0(1), 1(0), 2(1), 3(0)
    np.testing.assert_allclose(cov, [0.2, 0.4, 0.6, 0.8, 1.0])
    np.testing.assert_allclose(acc, [1, 1, 2 / 3, 3 / 4, 3 / 5])
    assert scoring.accuracy_at_coverage(correct, score, 1.0) == pytest.approx(correct.mean())
    assert scoring.accuracy_at_coverage(correct, score, 0.4) == 1.0
    rows = scoring.matched_coverage_comparison(correct, score, ~correct, score, coverages=(1.0,))
    assert rows[0]["acc_a"] == pytest.approx(0.6) and rows[0]["acc_b"] == pytest.approx(0.4)


# --------------------------------------------------------------- calibration


def test_brier_logloss_ece_hand_computed():
    p = np.array([0.1, 0.1, 0.9, 0.9])
    y = np.array([0, 1, 1, 1])
    assert scoring.brier(p, y) == pytest.approx((0.01 + 0.81 + 0.01 + 0.01) / 4)
    assert scoring.log_loss(p, y) == pytest.approx(-(np.log(0.9) + np.log(0.1) + 2 * np.log(0.9)) / 4)
    # bin 1: conf 0.1, acc 0.5, weight 1/2; bin 9: conf 0.9, acc 1.0, weight 1/2
    assert scoring.ece(p, y, n_bins=10) == pytest.approx(0.5 * 0.4 + 0.5 * 0.1)


# ----------------------------------------------------------- distributional


def test_interval_score_gneiting_raftery():
    assert scoring.interval_score(10, 20, 15, alpha=0.2) == pytest.approx(10.0)
    assert scoring.interval_score(10, 20, 5, alpha=0.2) == pytest.approx(10 + 10 * 5)
    assert scoring.interval_score(10, 20, 23, alpha=0.2) == pytest.approx(10 + 10 * 3)


def test_pinball_and_crps():
    assert scoring.pinball(0.9, 10, 12) == pytest.approx(1.8)
    assert scoring.pinball(0.9, 10, 8) == pytest.approx(0.2)
    # two-level hand example: 2 * mean(0.25*1, 0.25*1) = 0.5
    assert scoring.crps_from_quantiles([0.25, 0.75], [1.0, 3.0], 2.0) == pytest.approx(0.5)
    # point mass at m: mean over levels 0.01..0.99 of 2*pinball = 2 * mean(level) * |y-m| = |y-m|
    levels = scoring.default_levels(99)
    for y in (5.0, 1.0):
        crps = scoring.crps_from_quantiles(levels, np.full(99, 3.0), y)
        assert crps == pytest.approx(abs(y - 3.0), rel=1e-12)
        assert crps == pytest.approx(scoring.abs_error(3.0, y), rel=1e-12)
    batch = scoring.crps_from_quantiles(levels, np.full((2, 99), 3.0), np.array([5.0, 1.0]))
    np.testing.assert_allclose(batch, [2.0, 2.0])


def test_coverage_clopper_pearson():
    out = scoring.coverage([0] * 10, [1] * 10, [0.5] * 5 + [2.0] * 5)
    assert out["k"] == 5 and out["n"] == 10 and out["coverage"] == 0.5
    assert out["ci_lo"] == pytest.approx(0.187086, abs=1e-6)
    assert out["ci_hi"] == pytest.approx(0.812914, abs=1e-6)
    full = scoring.coverage([0] * 4, [1] * 4, [0.5] * 4)
    assert full["ci_hi"] == 1.0 and full["ci_lo"] == pytest.approx(0.025 ** (1 / 4))


# --------------------------------------------------------------- paired tests


def test_mcnemar_exact():
    # 8 rows A right/B wrong, 2 rows A wrong/B right, 5 both right, 3 both wrong
    a = np.array([1] * 8 + [0] * 2 + [1] * 5 + [0] * 3, dtype=bool)
    b = np.array([0] * 8 + [1] * 2 + [1] * 5 + [0] * 3, dtype=bool)
    r = scoring.mcnemar(a, b)
    assert (r["b"], r["c"]) == (8, 2)
    # P(X<=2 | Binom(10, .5)) = (1 + 10 + 45) / 1024
    assert r["p_two_sided"] == pytest.approx(2 * 56 / 1024)
    assert r["p_a_better"] == pytest.approx(56 / 1024)
    assert scoring.mcnemar(a, a)["p_two_sided"] == 1.0


def test_cluster_bootstrap_diff():
    # constant per-row improvement -> degenerate CI at the true diff
    b = np.linspace(0, 1, 30)
    clusters = np.repeat(np.arange(10), 3)
    diff, lo, hi = scoring.cluster_bootstrap_diff(b + 0.1, b, clusters, n_boot=500)
    assert diff == pytest.approx(0.1) and lo == pytest.approx(0.1) and hi == pytest.approx(0.1)

    # cluster-level effects: true diff 0.6 (6 of 10 clusters gain 1 on every row)
    gain = np.repeat((np.arange(10) < 6).astype(float), 3)
    a = np.zeros(30) + gain
    diff, lo, hi = scoring.cluster_bootstrap_diff(a, np.zeros(30), clusters, n_boot=2000, seed=0)
    assert diff == pytest.approx(0.6)
    assert lo < 0.6 < hi and 0.0 <= lo and hi <= 1.0
    # mean fast path agrees with the generic resampling path under the same seed
    generic = scoring.cluster_bootstrap_diff(a, np.zeros(30), clusters, n_boot=2000, seed=0, stat=lambda x: float(np.mean(x)))
    assert generic == pytest.approx((diff, lo, hi))
    # deterministic given seed
    assert scoring.cluster_bootstrap_diff(a, np.zeros(30), clusters, seed=0) == (diff, lo, hi)


# ---------------------------------------------------------- dataset scoring


def test_score_predictions_and_summarize():
    ex = {
        "id": "inv-1",
        "doc_type": "invoice",
        "template_id": "invoice/t1",
        "request": {
            "state": "Total $649.00",
            "fields": {"total": {"type": "number", "decimals": 2}, "po": {"type": "text"}, "eta": {"type": "estimate"}},
            "reference_date": "2026-10-09",
        },
        "gold": {
            "total": {"value": 649.0, "kind": "extract", "span": [6, 13]},
            "po": {"value": None, "kind": "null", "span": None},
            "eta": {"value": 12.0, "kind": "compute", "span": None},
        },
    }
    preds = {"inv-1": {
        "total": {"value": 649.0, "p_null": 0.0, "confidence": 1.0},
        "po": {"value": None, "p_null": 0.9},
        # eta prediction missing -> wrong
    }}
    rows = scoring.score_predictions([ex], preds)
    by_field = {r["field"]: r for r in rows}
    assert by_field["total"]["correct"] and by_field["po"]["correct"]
    assert not by_field["eta"]["correct"] and by_field["eta"]["pred_missing"]

    preds["inv-1"]["eta"] = {"value": 10.0, "median": 10.0, "p10": 8.0, "p90": 11.0, "quantiles": [10.0] * 99}
    eta = {r["field"]: r for r in scoring.score_predictions([ex], preds)}["eta"]
    assert eta["abs_error"] == 2.0 and eta["crps"] == pytest.approx(2.0)
    assert eta["interval_score_80"] == pytest.approx(3.0 + 2 / 0.2 * 1.0) and eta["covered_80"] is False

    s = scoring.summarize(rows)
    assert s["accuracy"] == pytest.approx(2 / 3)
    assert s["by_gold_kind"]["null"]["accuracy"] == 1.0
    assert s["null_detection"]["brier"] == pytest.approx((0.0 + 0.01) / 2)


# ------------------------------------------------------------ Clairic client


class _Resp:
    def __init__(self, status: int, body: dict, headers: dict | None = None):
        self.status_code = status
        self._body = body
        self.text = json.dumps(body)
        self.headers = headers or {}

    def json(self):
        return self._body


_REQUEST = {"state": "Total $649.00", "fields": {"total": {"type": "number"}}, "reference_date": "2026-10-09"}
_ANSWERS = {"total": {"type": "number", "value": 649.0, "p_null": 0.01, "confidence": 0.97}}


def _fake_post(responses):
    calls = []

    def post(url, json=None, headers=None, timeout=None):
        calls.append({"url": url, "json": json, "headers": headers})
        return responses[len(calls) - 1]

    return post, calls


def test_client_envelope_unwrap_and_cache(monkeypatch, tmp_path):
    envelope = {"result": {"model": "clairic", "answers": _ANSWERS, "truncated": False}, "success": True, "errors": []}
    post, calls = _fake_post([_Resp(200, envelope)])
    monkeypatch.setattr(requests, "post", post)
    cache = tmp_path / "cache.jsonl"
    client = clairic.ClairicClient("acct", "tok", cache)
    out = client.run(_REQUEST)
    assert out["answers"] == _ANSWERS
    assert calls[0]["url"] == "https://api.cloudflare.com/client/v4/accounts/acct/ai/run/@cf/cloudflare/clairic"
    body = calls[0]["json"]
    assert body["model"] == "clairic" and body["reference_date"] == "2026-10-09"
    assert body["options"] == {"include_quantiles": True, "return_state_text": False}
    assert calls[0]["headers"]["Authorization"] == "Bearer tok"

    # same client and a fresh client reading the cache file: no further network calls
    assert client.run(_REQUEST)["answers"] == _ANSWERS
    assert clairic.ClairicClient("acct", "tok", cache).run_many([{"id": "x", "request": _REQUEST}]) == {"x": _ANSWERS}
    assert len(calls) == 1
    entry = json.loads(cache.read_text(encoding="utf-8").strip())
    assert set(entry) == {"key", "request", "response", "fetched_at", "latency_s"}


def test_client_bare_response_and_failed_envelope(monkeypatch, tmp_path):
    post, _ = _fake_post([_Resp(200, {"model": "clairic", "answers": _ANSWERS}),
                          _Resp(200, {"result": None, "success": False, "errors": [{"message": "bad"}]})])
    monkeypatch.setattr(requests, "post", post)
    client = clairic.ClairicClient("acct", "tok", tmp_path / "c.jsonl")
    assert client.run(_REQUEST)["answers"] == _ANSWERS
    with pytest.raises(clairic.ClairicError, match="bad"):
        client.run(_REQUEST, include_quantiles=False)  # different cache key -> second call


def test_client_retries_on_429_then_succeeds(monkeypatch, tmp_path):
    sleeps = []
    monkeypatch.setattr(clairic.time, "sleep", sleeps.append)
    post, calls = _fake_post([_Resp(429, {}, {"Retry-After": "2"}), _Resp(503, {}), _Resp(200, {"answers": _ANSWERS})])
    monkeypatch.setattr(requests, "post", post)
    client = clairic.ClairicClient("acct", "tok", tmp_path / "c.jsonl", backoff_base=0.5)
    assert client.run(_REQUEST)["answers"] == _ANSWERS
    assert len(calls) == 3 and sleeps == [2.0, 1.0]  # Retry-After honored, then 0.5 * 2**1


def test_client_gives_up_and_does_not_retry_4xx(monkeypatch, tmp_path):
    monkeypatch.setattr(clairic.time, "sleep", lambda s: None)
    post, calls = _fake_post([_Resp(429, {})] * 3)
    monkeypatch.setattr(requests, "post", post)
    with pytest.raises(clairic.ClairicError, match="giving up"):
        clairic.ClairicClient("a", "t", tmp_path / "c.jsonl", max_retries=2).run(_REQUEST)
    assert len(calls) == 3
    post, calls = _fake_post([_Resp(400, {"errors": ["schema"]})])
    monkeypatch.setattr(requests, "post", post)
    with pytest.raises(clairic.ClairicError, match="HTTP 400"):
        clairic.ClairicClient("a", "t", tmp_path / "d.jsonl").run(_REQUEST)
    assert len(calls) == 1


def test_quantile_normalization():
    q = [float(i) for i in range(99)]
    assert clairic.normalize_quantiles(q) == q
    assert clairic.normalize_quantiles({f"p{i}": float(i) for i in range(99, 0, -1)}) == [float(i) for i in range(1, 100)]
    assert clairic.normalize_quantiles({str(i / 100): float(i) for i in range(1, 100)}) == [float(i) for i in range(1, 100)]
    ans = clairic.normalize_answers({"eta": {"value": 3.0, "percentiles": q}})
    assert ans["eta"]["quantiles"] == q and "percentiles" not in ans["eta"]
    with pytest.raises(clairic.ClairicError):
        clairic.normalize_quantiles([1.0, 2.0])


def test_relaxed_text_match_follows_squad_normalization():
    from extraction.scoring import relaxed_correct, squad_normalize
    spec = {"type": "text"}
    assert squad_normalize("The  most cost-efficient bidder.") == "most costefficient bidder"
    gold = {"value": "a quantity surveyor", "kind": "extract"}
    assert relaxed_correct(spec, gold, False, {"value": "Quantity surveyor."})
    assert not relaxed_correct(spec, gold, False, {"value": "surveyor"})
    # null stays an answer: relaxed never turns a wrong null into a right one
    assert not relaxed_correct(spec, gold, False, {"value": None})
    # non-text types keep strict correctness
    assert not relaxed_correct({"type": "number"}, {"value": 5.0}, False, {"value": "5.0."})
