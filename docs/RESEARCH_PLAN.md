# Research plan: an open single-pass typed extraction model vs Clairic

Living document. Update the **Status** and **Decision log** sections whenever a stage finishes or a
choice changes. Anything not written here should be assumed lost between chats.

## Goal and claim

Build a single-pass typed extraction model by fine-tuning an open backbone, and show it is **more
accurate than Cloudflare's Clairic** (`@cf/cloudflare/clairic`, preview, hosted only) on a pre-declared
domain. We expect Clairic to be a larger model with better-calibrated probabilities; the claim we aim
for is accuracy, scoped to a domain we fix before measuring. Calibration and decision quality are reported
honestly alongside, even where Clairic wins.

Clairic facts (docs, 2026-10-19): $0.10/M input tokens; 65,536-token context; fields that do not fit next
to the state in one **8,192-token pass** are split into parallel passes; field types `text, number, date,
time, datetime, duration, list, estimate, forecast`; per-field `null_threshold` (default 0.5);
`options.include_quantiles` returns 99 percentiles for estimates. Weights, recipe and base model are not
published ("open weights soon").

Public prior (Cloudflare's released decision model, not our own work): `Cloudflare/clef-flash` is
Qwen3.5-9B plus a small "joint schema head"; its input puts the state first, then one block per schema
field, and the head cross-attends from field queries to the backbone's last hidden states, with a small
transformer over the fields. `Cloudflare/clef` is a 27B Qwen backbone. We read Clairic as likely the same
family [inference]. The user's earlier decision-model project is deliberately excluded from this work.

## Model design

Request `(x, S)`: document `x`, schema `S = {(name, type, instructions, constraints)}`. Output a typed
answer per field in Clairic's response shape (`value, span, p_null, confidence, guess, median, p10, p90,
quantiles`), so one scorer handles both models.

- **Backbone**: Qwen3.5-4B (start), frozen in stage 0, LoRA later. 32 layers: 3 Gated DeltaNet (linear)
  : 1 full attention (full attention after layers 4, 8, ..., 32, 1-based), hidden 2560, built-in vision
  tower (unused until images). Fallback for diagnosis: Qwen3-4B (all full attention).
- **Layout** (`extraction/encode.py`): system prompt, `STATE:` + document, `SCHEMA FIELDS:` + one block
  per field (ID, TYPE, constraints, INSTRUCTION), assistant suffix. Document first, so its hidden states
  are schema-independent and cacheable; field conditioning enters through field queries.
- **Field query**: mean over instruction tokens + last token of the field block + last token of the
  sequence.
- **Heads** (planned, by stage):
  - null: `sigmoid(MLP(q))`, log loss.
  - span pointer: start/end bilinear scores over state tokens, decode best (s, e) with e - s < 48 tokens.
  - number/date: gate over copy (span + deterministic parser), compute (operation + operand pointers,
    executed in code, operands may reference other fields), estimate (monotone quantile function).
  - estimate/forecast: monotone quantile function trained with CRPS (pinball integrated over levels).
  - joint record: field mixer (bidirectional layers over field queries); later a Gaussian copula with
    low-rank covariance trained with the energy score (beta in (0, 2)).
- **Loss principle**: strictly proper scores only (log loss for discrete choices, marginalized over all
  programs/spans that produce the gold value; CRPS for numeric values, not log density).

## Stages and gates

| Stage | What | Gate to continue |
|---|---|---|
| 0 | Frozen Qwen3.5-4B; one span+null probe per layer; `notebooks/stage0_probe_colab.ipynb` | Best layer's extract accuracy on unseen-template synth and SQuAD dev clearly above a trivial baseline; decide which layer(s) to read. If linear-attention weakness shows (FDA low, full-attention layers much better), probe Qwen3-4B with the same notebook |
| 1 | Heads only (field mixer + span/null + number gate), backbone frozen | Beats the stage-0 probe at the chosen layer |
| 2 | LoRA (rank 16-32; capacity estimate: one rank unit on an 8B-class model ~ 5M bits at 2 bits/param, data ~ 5e7 bits) + heads, 8k tokens, activation recompute | Beats stage 1 on held-out suites |
| 3 | Compute fields (operation DAG), estimates (quantile head), lists; optional latent refinement loop | Per-type metrics improve; calibration post-hoc (temperature, conformalized quantile regression) |
| 4 | Head-to-head vs Clairic on the declared domain (below) | Pre-registered test |

## Evaluation protocol (vs Clairic)

- **Same requests**: identical `state`, `fields` (incl. bounds/step/scale), `reference_date`, images.
  Clairic called through `extraction/clairic.py` (cached JSONL, timestamps). Rerun a fixed ~200-item canary
  set before each comparison to detect silent model updates.
- **Primary metric (accuracy)**: exact match after normalization with **null counted as an answer**;
  estimates/forecasts: absolute error of the reported median.
- **Abstention control**: compare accuracy at matched coverage (sweep each model's `p_null`/`confidence`
  threshold) and report the risk-coverage curve; a model must not win by answering null more often.
- **Test**: one-sided paired McNemar on exact match; document-clustered bootstrap CIs. Sample size:
  10% discordant pairs, detect +3 pp at 80% power, alpha 0.05 one-sided: ~685 fields (2 pp: ~2,000).
- **Secondary (reported, not the claim)**: Brier/log loss of `p_null`, ECE of P(correct), CRPS and
  interval score (alpha = 0.2) for estimates, coverage with Clopper-Pearson CI (coverage alone is improper),
  cost-vs-threshold decision curves, latency and cost.
- **Data ranking for the claim**: (1) private or freshly collected, human-labelled documents in the declared
  domain; (2) prospective estimates/forecasts scored after outcomes arrive; (3) public benchmarks
  (possible contamination in Clairic's training); (4) our synthetic templates (diagnostic only, biased to us).
- Report raw score differences, not skill-score ratios (improper in general).
- **Domain**: TBD; must be fixed and written here before stage 4 data is scored.

## Data

| Source | Use | Notes |
|---|---|---|
| `extraction/synth.py` | train + eval (template-held-out) | invoices, receipts, support emails, shipping notices; 4 templates each; ~23% null (decoy-weighted), ~15% compute, exact spans |
| SQuAD 2.0 | train + eval | span + unanswerable; questions per paragraph form one multi-field record |
| Gretel synthetic PII finance | train + eval | entity spans; single-surface, single-line labels only |
| FDA (`hazyresearch/based-fda`) | eval | long documents; the extraction task where linear attention lags |
| DROP, TAT-QA | stage 3 | computed numbers/dates |
| CORD v2 | stage 3+ | receipt images |
| `drew-ipp/invoice-extraction-benchmark` | eval | 181 synthetic invoices with answer keys |

## Infrastructure

- Training/eval on a Colab A100 80 GB; repo `github.com/pherber3/extraction-model` (public) cloned in the
  notebook. Colab installs `transformers>=5.20`, `datasets`, `flash-linear-attention` with pip.
- Local: uv project (`uv add`, never `uv pip install`), CPU torch; `DRY_RUN=True` runs the notebook on a
  random 4-layer Qwen3.5 in ~20 s.

## Risks

- Hybrid linear attention may weaken exact-span extraction on structured, repetitive documents (Physics of
  LMs 4.1 FDA/SWDE gap; recurrent-state write sparsity). Stage 0 measures it; Qwen3-4B is the fallback.
- Field isolation: block-diagonal masks do not apply to DeltaNet layers; currently fields share one causal
  sequence with shuffled order. Later: snapshot the recurrent state after the document and fork per field.
- Synthetic-to-real gap; contamination of public sets in Clairic's training; Clairic version drift.

## Status

- 2026-10-19: stage 0 code complete and verified on CPU (64 tests; notebook DRY_RUN end to end; probes
  overfit 60 docs from 0 to ~57% on a random tiny backbone). First real stage-0 run started on Colab A100.

## Decision log

- 2026-10-19: Qwen3.5-4B as the first backbone (user choice); Colab A100 only (no local AMD/WSL setup).
- 2026-10-19: Claim = accuracy on a declared domain, at matched coverage; calibration reported, not claimed.
- 2026-10-19: Stage 0 scope = single-span types + null; compute fields train the null head only.
- 2026-10-19: Public datasets adapted rather than generating everything; own generator kept for
  Clairic-format typed fields, decoy nulls and compute fields.
