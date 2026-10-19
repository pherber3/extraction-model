# extraction-model

A single-pass typed extraction model in the style of Cloudflare's Clairic (`@cf/cloudflare/clairic`),
built by fine-tuning an open backbone (Qwen3.5-4B), plus a harness that scores it and Clairic on the
same requests.

One forward pass reads the request: the document first, then one block per schema field. Small heads
read the backbone's hidden states and return Clairic-shaped answers (`value`, `span`, `p_null`,
`confidence`, ...), so `extraction.scoring` treats both models identically.

## Layout

| Path | Contents |
|---|---|
| `extraction/encode.py` | Request → token sequence; char↔token span mapping |
| `extraction/heads.py` | Span pointer + null head on one layer; span decoding |
| `extraction/probe.py` | Stage 0: frozen backbone, one probe per layer, predictions in Clairic format |
| `extraction/synth.py`, `extraction/normalize.py` | Synthetic invoices / receipts / support emails / shipping notices with exact spans, null decoys and computed fields; the shared value normalizer |
| `extraction/public_data.py` | SQuAD 2.0, Gretel synthetic finance (train/eval) and FDA (eval) converted to the same format |
| `extraction/scoring.py` | Exact match (null counts as an answer), matched-coverage accuracy, Brier/ECE, CRPS, interval score, McNemar, document-clustered bootstrap |
| `extraction/clairic.py` | Cached REST client for Clairic (`CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN`) |
| `notebooks/stage0_probe_colab.ipynb` | Stage 0 on a Colab A100 |

## Example format

One JSON object per example (`synth.write_jsonl` / `read_jsonl`):

```json
{"id": "inv-000123", "doc_type": "invoice", "template_id": "invoice/t3",
 "request": {"state": "...", "fields": {"total": {"type": "number", "decimals": 2, "unit": "USD"}}, "reference_date": "2026-10-09"},
 "gold": {"total": {"value": 1080.0, "kind": "extract", "span": [212, 221]}}}
```

`kind` is `extract` (read from `span`), `compute` (derived; no span) or `null` (absent from the document).

## Local development (CPU)

Dependencies are managed with uv (`uv add <pkg>`; torch comes from the CPU wheel index).

```sh
uv sync
uv run pytest -q
```

The notebook also runs locally: set `DRY_RUN = True` to use a randomly initialised 4-layer Qwen3.5 with
the real tokenizer and small data. It exercises every cell in about 20 s.

## Stage 0 on Colab

1. The repo is private (`github.com/pherber3/extraction-model`). In Colab, open the Secrets panel, add
   `GITHUB_TOKEN` (a token with read access to this repo) and enable notebook access; the setup cell
   clones with it.
2. Open the notebook in Colab, choose an A100 runtime, Run all. The setup cell installs
   `transformers>=5.20`, `datasets` and `flash-linear-attention` with pip (Colab is not a uv environment).
3. Results land in `/content/stage0_results` (a per-layer table, the best layer's per-field rows and the
   probe weights) and are downloaded as a zip; `USE_DRIVE = True` also copies them to Drive.

What it measures: for each of the 32 layers, span accuracy and null detection on four held-out suites:
synthetic documents from templates never seen in training, SQuAD 2.0 dev, Gretel test and FDA.
Qwen3.5-4B interleaves 3 Gated DeltaNet layers with 1 full-attention layer, so the table marks
which layers are full attention.

## Status

- Verified on CPU: tests (encoder offsets, synthetic data round-trips, scoring math, Clairic client
  against a fake API), and the notebook end to end in `DRY_RUN`. With the random tiny backbone the
  probes overfit 60 training documents from 0 to about 57% accuracy, so the training path learns.
- Not yet run: the notebook on a real Qwen3.5-4B, and the Clairic client against the live API.
