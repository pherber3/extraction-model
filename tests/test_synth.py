import datetime as dt
import random
import time

import pytest

from extraction import synth
from extraction.normalize import normalize_value, text_equal

REF = "2026-10-09"


@pytest.fixture(scope="module")
def examples_with_nulls():
    return synth.generate_with_nulls(500, seed=0)


@pytest.mark.parametrize(
    ("type_", "raw", "decimals", "want"),
    [
        ("number", "$1,080.00", 2, 1080.0),
        ("number", "USD 1080", 2, 1080.0),
        ("number", "1.080,00 EUR", 2, 1080.0),
        ("number", "EUR 12.345,67", 2, 12345.67),
        ("number", "52,10 €", 2, 52.1),
        ("number", "£1,234,567.89", 2, 1234567.89),
        ("number", "-$5.00", 2, -5.0),
        ("number", "2.4 kg", 1, 2.4),
        ("number", "3", 0, 3),
        ("duration", "30 days", 0, 30),
        ("date", "Sept 28, 2026", None, "2026-09-28"),
        ("date", "28/09/2026", None, "2026-09-28"),
        ("date", "09/28/2026", None, "2026-09-28"),
        ("date", "03/04/2026", None, "2026-04-03"),
        ("date", "28.09.2026", None, "2026-09-28"),
        ("date", "September 28", None, "2026-09-28"),
        ("date", "Mon, Sept 28", None, "2026-09-28"),
        ("date", "November 3rd", None, "2026-11-03"),
        ("date", "Wednesday, October 7, 2026", None, "2026-10-07"),
        ("time", "2:32 PM", None, "14:32"),
        ("time", "12:05am", None, "00:05"),
        ("datetime", "Sept 28, 2026 at 2:30 PM", None, "2026-09-28T14:30"),
    ],
)
def test_normalize_value(type_, raw, decimals, want):
    got = normalize_value(type_, raw, decimals, REF)
    assert got == want and type(got) is type(want)


def test_normalize_rejects_garbage():
    for type_, raw in [("number", "n/a"), ("date", "soon"), ("time", "later"), ("date", "Sept 28")]:
        with pytest.raises(ValueError):
            normalize_value(type_, raw)  # "Sept 28" has no year and no reference date


def test_text_equal():
    assert text_equal("  Northwind Supply Co. ", "northwind supply co")
    assert not text_equal("Dana", "Diana")


def test_extract_spans_round_trip(examples_with_nulls):
    n_checked = 0
    for ex, _ in examples_with_nulls:
        state, ref = ex["request"]["state"], ex["request"]["reference_date"]
        for name, g in ex["gold"].items():
            if g["kind"] != "extract":
                continue
            spec = ex["request"]["fields"][name]
            if spec["type"] == "list":
                pairs = list(zip([spec["items"]] * len(g["value"]), g["spans"], g["value"], strict=True))
            else:
                pairs = [(spec["type"], g["span"], g["value"])]
            for type_, (s, e), value in pairs:
                got = normalize_value(type_, state[s:e], spec.get("decimals"), ref)
                want = normalize_value(type_, value, spec.get("decimals"), ref)
                assert text_equal(got, want) if type_ == "text" else got == want, (ex["id"], name, state[s:e])
                n_checked += 1
    assert n_checked > 2000


def test_gold_value_types(examples_with_nulls):
    for ex, _ in examples_with_nulls:
        for name, g in ex["gold"].items():
            spec = ex["request"]["fields"][name]
            if g["kind"] == "null":
                assert g["value"] is None and g["span"] is None and g["spans"] is None
                continue
            v = g["value"]
            if spec["type"] == "date":
                dt.date.fromisoformat(v)
            elif spec.get("integer"):
                assert type(v) is int
            elif spec["type"] == "number":
                assert isinstance(v, float) and round(v, spec["decimals"]) == v
            if g["kind"] == "compute":
                assert g["span"] is None and g["spans"] is None


def test_null_values_absent_from_document(examples_with_nulls):
    n_null = 0
    for ex, nulls in examples_with_nulls:
        assert set(nulls) == {n for n, g in ex["gold"].items() if g["kind"] == "null"}
        for name, check in nulls.items():
            assert synth.null_is_absent(ex["request"]["state"], check, ex["request"]["reference_date"]), (ex["id"], name)
            n_null += 1
    assert n_null > 300


def test_kind_rates_and_template_coverage(examples_with_nulls):
    kinds = [g["kind"] for ex, _ in examples_with_nulls for g in ex["gold"].values()]
    null_rate = kinds.count("null") / len(kinds)
    compute_rate = kinds.count("compute") / len(kinds)
    assert 0.20 <= null_rate <= 0.30
    assert 0.10 <= compute_rate <= 0.20
    assert {ex["template_id"] for ex, _ in examples_with_nulls} == set(synth.TEMPLATE_IDS)
    assert all(len([t for t in synth.TEMPLATE_IDS if t.startswith(d + "/")]) >= 4 for d in synth.DOC_TYPES)


def _money(g, name):
    e = g.entries.get(name)
    return 0.0 if e is None or e["value"] is None else e["value"]


@pytest.mark.parametrize("doc_type", synth.DOC_TYPES)
def test_compute_values_are_exact(doc_type):
    rng = random.Random(7)
    for _ in range(200):
        rec, _, _, g, _ = synth._render(rng, doc_type)
        e = g.entries
        if doc_type == "invoice":
            want = _money(g, "subtotal") - _money(g, "discount") + _money(g, "tax") + _money(g, "shipping")
            assert e["total"]["value"] == pytest.approx(want, abs=1e-9)
            if e["due_date"]["kind"] == "compute":
                issue = dt.date.fromisoformat(e["issue_date"]["value"])
                assert dt.date.fromisoformat(e["due_date"]["value"]) - issue == dt.timedelta(e["days_until_due"]["value"])
            assert e["item_count"]["value"] == len(e["line_items"]["value"])
        elif doc_type == "receipt":
            want = _money(g, "subtotal") + _money(g, "tax") + _money(g, "tip")
            assert e["total"]["value"] == pytest.approx(want, abs=1e-9)
        elif doc_type == "support_email":
            want = _money(g, "price_paid") + _money(g, "tax") + _money(g, "shipping_cost")
            assert e["total_paid"]["value"] == pytest.approx(want, abs=1e-9)
            if "days_to_delivery" in e:
                days = dt.date.fromisoformat(e["delivery_date"]["value"]) - dt.date.fromisoformat(e["order_date"]["value"])
                assert e["days_to_delivery"]["value"] == days.days
        else:
            if "transit_days" in e:
                days = dt.date.fromisoformat(e["estimated_delivery"]["value"]) - dt.date.fromisoformat(e["ship_date"]["value"])
                assert e["transit_days"]["value"] == days.days
            if "days_until_delivery" in e:
                days = dt.date.fromisoformat(e["estimated_delivery"]["value"]) - rec["ref"]
                assert e["days_until_delivery"]["value"] == days.days


def test_deterministic_per_seed():
    a = synth.generate(60, seed=3)
    assert a == synth.generate(60, seed=3)
    assert a[:20] == synth.generate(20, seed=3)
    assert a != synth.generate(60, seed=4)


def test_split_by_template_is_disjoint():
    exs = synth.generate(200, seed=1)
    hold = {"invoice/t1", "receipt/t3"}
    train, held = synth.split_by_template(exs, hold)
    assert len(train) + len(held) == len(exs)
    assert {e["template_id"] for e in held} <= hold
    assert not {e["template_id"] for e in train} & hold


def test_generate_500_is_fast():
    t0 = time.perf_counter()
    synth.generate(500, seed=0)
    assert time.perf_counter() - t0 < 5.0
