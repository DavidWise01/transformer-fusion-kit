# v2 — the shadow-channel rebuild, and what an honest harness found

The v1 teardown reported a measured record: three-body fusion at **25.02** test ppl,
with the **shadow gate buying −1.21** over static mixing, and the datastore said to
**halve** the transformer's perplexity. v2 rebuilt the evaluation to be leak-free and
added a witness channel and a checkpoint fingerprint. Running it corrected two of those
three claims. This file is the audit trail.

## What changed in the code

### `fusion.py` — leak removed, gate kept honest
- **No test-set tuning.** v1 grid-searched the mix weights on the *same* test slice it
  reported them on (a leak). v2 tunes on a validation slice (front 20%) and reports on
  the held-out remainder.
- **Input-integrity guard.** A non-finite entry in any channel used to make `pv < best`
  always-False, so a poisoned run silently returned "weights None". The guard now names
  the corrupt channel and clamps it to a floor instead of fusing blind.
- **`rel` scaling fixed.** The old `exp(-d_nn/0.2/d_nn.mean())` saturated to ≈0 at the
  mean distance, so the adaptive term did almost nothing; now scaled by the distance
  spread to span (0,1].
- **Diagnostic.** Prints exactly what the adaptive `rel` term buys over the best static
  blend — so the gate can no longer hide behind a leak.
- (Carried from v1: the kNN softmax **row-min shift** that stops the 512-dim underflow.
  Without it the datastore dies — 97% NaN, recall 1% — and the whole comparison is moot.)

### `shadow.py` — a witness, not a mixer (NEW)
Independently recomputes the KN and kNN bodies **from their raw inputs** (counts,
datastore), diffs them against the primary pipeline's values, and folds channel 2 into
channel 1 **conditioned on agreement** — refusing a channel it can't verify instead of
blending poison through. Returns an audit report, not just a number.

### `verify.py` — offline proof of the witness (NEW, no GPU)
Drives the real `shadow.*` on synthetic channels engineered to reproduce the run-log
faults. **Verified here (CPU):**
- clean → gate and shadow both work;
- NaN poison (5% of positions) → old gate returns `weights None`; **shadow flags the
  channel (nonfinite=True)** and still produces a finite, usable fusion;
- silent KN drift (misaligned by `order`, finite but wrong) → old gate passes it through
  silently; **shadow flags 99.8% of positions** via the independent recompute.

### `model_id.py` — checkpoint identity as a fact on disk (NEW)
ARCH / WEIGHTS / STATE fingerprints (sha256 of shape-signature, of tensor bytes, of
step/best-val). Verdicts: IDENTICAL / SAME MODEL DIFFERENT TRAINING / DIFFERENT MODEL.
**Run on the real checkpoint** (`transformer.pt.fingerprint.json`): deterministic across
runs, and it surfaced that the checkpoint carries **76,552,192 params**, not the "51M"
the v1 teardown stated (untied embeddings) — the first thing the fingerprint corrected.

## What the honest re-run measured (RTX 4050, WT-103 test, 200k positions, 1M-key datastore)

| body / method                         | v1 (reported)            | v2 (leak-free, held-out)     |
|---------------------------------------|--------------------------|------------------------------|
| transformer                           | 52.45                    | 53.37                        |
| Kneser-Ney counts                     | 282.61                   | 285.26                       |
| best static blend (val-tuned)         | (26.23, hand-set 0.6/0.3/0.1) | **25.25**  (0.7 / 0.3 / ~0) |
| + reliability gate                    | **25.02**                | **25.24**                    |
| adaptive gate over static             | **−1.21** (claimed)      | **+0.009 ppl** (measured)    |

## The two corrections

1. **The −1.21 "shadow-gate win" was a test-slice tuning leak.** It compared a *hand-set*
   static blend against a gate *tuned on the slice it was scored on*. Tuned honestly on
   validation, the gate buys **+0.009 ppl** — the diagnostic's own words: "the gate's win
   is essentially the static term; rel barely contributes."
2. **The datastore is not doing the halving.** At the 1M-key cap (2.5% of ~40M, recall
   36.9%) the honest search puts **~0 weight on kNN**; transformer(0.7)+KN(0.3) alone
   already lands 25.25. The 53→25 drop is the *count-model mixture*, not the datastore.
   (kNN-LM's published −2 to −4 needs the full ~103M-key store — not a claim this cap can
   make.)

**What holds:** the fusion record itself (~25.2) survives, now measured on held-out data
instead of the slice it was tuned on. **What was retracted:** the gate's −1.21 and the
datastore's "halving," both artifacts of an evaluation that graded its own homework. The
value of v2 is not a lower number — it is a *truer* one, plus a witness and a fingerprint
that make the next number harder to fool.
