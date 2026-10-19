import pytest

from extraction.encode import char_span_to_tokens, encode_example, token_span_to_chars

transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def tokenizer():
    return transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")


STATE = "Hi, order A-88412 arrived on Sept 28, 2026 with a cracked screen. I paid $649.00 plus $52.10 tax."


@pytest.mark.parametrize("surface", ["A-88412", "Sept 28, 2026", "$649.00", "cracked screen", "52.10"])
def test_gold_span_round_trips_through_tokens(tokenizer, surface):
    start = STATE.index(surface)
    offsets = tuple(tuple(p) for p in tokenizer(STATE, add_special_tokens=False, return_offsets_mapping=True).offset_mapping)
    first, last = char_span_to_tokens(offsets, start, start + len(surface))
    char_start, char_end = token_span_to_chars(STATE, offsets, first, last)
    # Token boundaries may extend past the surface text (e.g. a merged "$52" token); they must cover it.
    assert char_start <= start and char_end >= start + len(surface)
    assert surface in STATE[char_start:char_end]


def test_field_positions_point_inside_schema_and_state_precedes_it(tokenizer):
    example = {
        "id": "x",
        "request": {
            "state": STATE,
            "fields": {
                "order_id": {"type": "text", "instructions": "Order number"},
                "tracking_number": {"type": "text"},
            },
        },
        "gold": {
            "order_id": {"value": "A-88412", "kind": "extract", "span": [10, 17]},
            "tracking_number": {"value": None, "kind": "null", "span": None},
        },
    }
    record = encode_example(tokenizer, example)
    state_start, state_end = record.state_range
    decoded_state = tokenizer.decode(record.input_ids[state_start:state_end])
    assert decoded_state == STATE
    order, tracking = record.fields
    assert tokenizer.decode(record.input_ids[slice(*order.instruction_span)]) == "Order number"
    assert tokenizer.decode(record.input_ids[slice(*tracking.instruction_span)]) == "tracking_number"
    assert all(f.instruction_span[0] >= state_end for f in record.fields)
    assert tokenizer.decode(record.input_ids[state_start + order.start_token : state_start + order.end_token + 1]).strip() == "A-88412"
    assert (tracking.start_token, tracking.end_token) == (-1, -1)


def test_truncation_drops_spans_that_fall_outside(tokenizer):
    long_state = "filler " * 400 + "Order A-88412."
    start = long_state.index("A-88412")
    example = {
        "id": "y",
        "request": {"state": long_state, "fields": {"order_id": {"type": "text"}}},
        "gold": {"order_id": {"value": "A-88412", "kind": "extract", "span": [start, start + 7]}},
    }
    record = encode_example(tokenizer, example, max_length=200)
    assert record.truncated
    assert len(record.input_ids) <= 200
    assert record.fields[0].start_token == -1
