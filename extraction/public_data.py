"""Adapters from public span-labelled datasets to the shared example format.

None of these datasets matches Clairic's typed schema, so each adapter keeps only what it labels
exactly. Everything is a ``text`` field, because the sources give surface strings, not typed values:

- SQuAD 2.0 (CC BY-SA 4.0): Wikipedia paragraphs; each question becomes one field, unanswerable
  questions become null fields. All questions about one paragraph form one multi-field record.
- Gretel synthetic PII finance (Apache-2.0): LLM-written financial documents with labelled entity
  spans. A label becomes a field only when all its mentions share one single-line surface string
  (so the field has one right answer; about 0.7% of spans spill across lines and are dropped);
  labels absent from the document become null fields.
- FDA (Hazy Research ``based-fda``): key/value pairs from FDA 510(k) summaries; evaluation only.
"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from typing import Any

_FIELD_NAME = re.compile(r"[^A-Za-z0-9_]+")


def _field_name(text: str, taken: set[str]) -> str:
    base = _FIELD_NAME.sub("_", text.strip().lower()).strip("_")[:56] or "field"
    if base[0].isdigit():
        base = f"f_{base}"
    name, suffix = base, 2
    while name in taken:
        name, suffix = f"{base}_{suffix}", suffix + 1
    taken.add(name)
    return name


def squad_v2(split: str = "train", limit: int | None = None, seed: int = 0) -> list[dict[str, Any]]:
    from datasets import load_dataset

    rows = load_dataset("rajpurkar/squad_v2", split=split)
    by_context: dict[str, list[dict[str, Any]]] = defaultdict(list)
    titles: dict[str, str] = {}
    for row in rows:
        by_context[row["context"]].append(row)
        titles[row["context"]] = row["title"]
    contexts = sorted(by_context)
    random.Random(seed).shuffle(contexts)
    examples = []
    for index, context in enumerate(contexts[:limit]):
        fields, gold = {}, {}
        for number, row in enumerate(by_context[context], start=1):
            name = f"q{number}"
            fields[name] = {"type": "text", "instructions": row["question"]}
            if row["answers"]["text"]:
                start = row["answers"]["answer_start"][0]
                text = row["answers"]["text"][0]
                gold[name] = {"value": text, "kind": "extract", "span": [start, start + len(text)]}
            else:
                gold[name] = {"value": None, "kind": "null", "span": None}
        examples.append(
            {
                "id": f"squad-{split}-{index:06d}",
                "doc_type": "wikipedia",
                "template_id": f"squad/{titles[context]}",
                "request": {"state": context, "fields": fields},
                "gold": gold,
            }
        )
    return examples


GRETEL_LABEL_INSTRUCTIONS = {
    "company": "Name of the company",
    "name": "Full name of the person",
    "date": "The date mentioned",
    "street_address": "Street address",
    "email": "Email address",
    "phone_number": "Phone number",
    "iban": "IBAN",
    "account_pin": "Account PIN",
    "swift_bic_code": "SWIFT/BIC code",
    "bban": "Bank account number (BBAN)",
    "credit_card_number": "Credit card number",
    "customer_id": "Customer ID",
    "employee_id": "Employee ID",
    "ssn": "Social security number",
}


def gretel_finance(split: str = "train", limit: int | None = None, seed: int = 0, null_fields: int = 2) -> list[dict[str, Any]]:
    from datasets import load_dataset

    rows = load_dataset("gretelai/synthetic_pii_finance_multilingual", split=split)
    rng = random.Random(seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    examples = []
    for index in order:
        if limit is not None and len(examples) >= limit:
            break
        row = rows[index]
        if row["language"] != "English":
            continue
        text = row["generated_text"]
        surfaces: dict[str, set[str]] = defaultdict(set)
        first_span: dict[str, list[int]] = {}
        for span in json.loads(row["pii_spans"]):
            surfaces[span["label"]].add(text[span["start"] : span["end"]])
            first_span.setdefault(span["label"], [span["start"], span["end"]])
        # One surface string per label, and no line breaks (multi-line spans are annotation spill-over).
        unique = [
            label
            for label, values in surfaces.items()
            if len(values) == 1 and label in GRETEL_LABEL_INSTRUCTIONS and "\n" not in next(iter(values))
        ]
        absent = [label for label in GRETEL_LABEL_INSTRUCTIONS if label not in surfaces]
        chosen = unique + rng.sample(absent, min(null_fields, len(absent)))
        if not unique:
            continue
        rng.shuffle(chosen)
        fields, gold, taken = {}, {}, set()
        for label in chosen:
            name = _field_name(label, taken)
            fields[name] = {"type": "text", "instructions": GRETEL_LABEL_INSTRUCTIONS[label]}
            if label in first_span:
                start, end = first_span[label]
                gold[name] = {"value": text[start:end], "kind": "extract", "span": [start, end]}
            else:
                gold[name] = {"value": None, "kind": "null", "span": None}
        examples.append(
            {
                "id": f"gretel-{split}-{row['index']}",
                "doc_type": "finance_synthetic",
                "template_id": f"gretel/{row['document_type']}",
                "request": {"state": text, "fields": fields},
                "gold": gold,
            }
        )
    return examples


def fda(limit: int | None = None) -> list[dict[str, Any]]:
    """Evaluation set: one record per document, one field per key whose value occurs in the text."""
    from datasets import load_dataset

    rows = load_dataset("hazyresearch/based-fda", split="validation")
    documents: dict[str, dict[str, Any]] = {}
    for row in rows:
        document = documents.setdefault(row["doc_id"], {"text": row["text"], "pairs": []})
        document["pairs"].append((row["key"], row["value"]))
    examples = []
    for doc_id, document in sorted(documents.items())[:limit]:
        text = document["text"]
        fields, gold, taken = {}, {}, set()
        for key, value in document["pairs"]:
            start = text.lower().find(value.lower())
            if start < 0:
                continue
            name = _field_name(key, taken)
            fields[name] = {"type": "text", "instructions": key}
            gold[name] = {"value": text[start : start + len(value)], "kind": "extract", "span": [start, start + len(value)]}
        if fields:
            examples.append(
                {
                    "id": f"fda-{doc_id}",
                    "doc_type": "fda_510k",
                    "template_id": "fda",
                    "request": {"state": text, "fields": fields},
                    "gold": gold,
                }
            )
    return examples
