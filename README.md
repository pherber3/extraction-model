# extraction-model

A single-pass typed extraction model in the style of Cloudflare's Clairic (`@cf/cloudflare/clairic`),
built by fine-tuning an open backbone (Qwen3.5-4B), plus a harness that scores it and Clairic on the
same requests.

Goals, stage gates, the evaluation protocol against Clairic and the decision log live in
[`docs/RESEARCH_PLAN.md`](docs/RESEARCH_PLAN.md).

One forward pass reads the request: the document first, then one block per schema field. Small heads
read the backbone's hidden states and return Clairic-shaped answers (`value`, `span`, `p_null`,
`confidence`, ...), so `extraction.scoring` treats both models identically.

## Layout

| Path | Contents |
|---|---|
| `extraction/encode.py` | Request → token sequence; char↔token span mapping |
| `extraction/heads.py` | Stage-0 span+null probe; stage-1 extraction head (layer mix, field mixer, start-conditioned end) |
| `extraction/train.py` | Frozen backbone: load/truncate, train any heads from one forward per batch, predict in Clairic format |
| `extraction/stage0.py`, `extraction/stage1.py` | Stage pipelines (data, training, per-suite evaluation, outputs) |
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

The stage pipelines live in `extraction/stage0.py` (one probe per layer) and `extraction/stage1.py` (the
stage-1 extraction head trained next to the stage-0 probe); `extraction/train.py` trains any heads on a
frozen backbone. `uv run python scripts/run_stage0.py --dry-run` (or `run_stage1.py --dry-run`) runs a
pipeline on CPU with a randomly initialised 4-layer Qwen3.5, the real tokenizer and small data (~20 s).
The notebook calls the stage-0 code (`DRY_RUN = True` for the same smoke test).

## Stage 0 on Colab

**From the terminal (Colab CLI).** The CLI is Linux/macOS-only and needs Python ≥3.13, so on Windows
it runs through `scripts/colab.py` in its own uvx environment (it stubs the Unix-only console module and
defaults to `--auth=adc`):

```sh
# once: gcloud auth application-default login --scopes="openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory"
alias colab='uvx --python 3.13 --from google-colab-cli python scripts/colab.py'
colab new -s stage0 --gpu A100                    # billable from here
colab upload -s stage0 results/code.zip /content/code.zip   # zip of the working tree
# then `colab exec` a launcher that unzips, pip-installs transformers>=5.20 datasets
# flash-linear-attention, and starts scripts/run_stage0.py in the background with output to a log;
# `colab exec` again to read the log; `colab download` the results; `colab stop -s stage0`.
```

**From the browser.** Open `notebooks/stage0_probe_colab.ipynb` from `github.com/pherber3/extraction-model`
(File → Open notebook → GitHub), choose an A100 runtime, Run all. The setup cell clones the repo and
pip-installs the same packages. Results go to `/content/stage0_results` and download as a zip; the probe
weights (~1 GB) are only included with `INCLUDE_HEADS = True`.

What it measures: for each of the 32 layers, span accuracy and null detection on four held-out suites:
synthetic documents from templates never seen in training, SQuAD 2.0 dev, Gretel test and FDA.
Qwen3.5-4B interleaves 3 Gated DeltaNet layers with 1 full-attention layer, so the table marks
which layers are full attention.

## Status

- Verified on CPU: tests (encoder offsets, synthetic data round-trips, scoring math, Clairic client
  against a fake API), and the notebook end to end in `DRY_RUN`. With the random tiny backbone the
  probes overfit 60 training documents from 0 to about 57% accuracy, so the training path learns.
- Not yet run: the notebook on a real Qwen3.5-4B, and the Clairic client against the live API.
