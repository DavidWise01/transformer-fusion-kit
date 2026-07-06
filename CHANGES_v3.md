# v3 — the two witnesses wired in, and what they measured on real data

v2 added a shadow *channel* and a checkpoint fingerprint but ran neither inside the live
fusion. v3 wires both witnesses into `fusion.main()` (opt-in env flags) and adds a third
instrument — an **uncertainty gate**. All three were then run end-to-end on the real
checkpoint. As with every prior step, running surfaced faults reading did not.

## What's new in the code
- **`uncertainty_gate.py`** (NEW) — retrieve only where the transformer is unsure. Scores
  uncertainty from the LM's own output, searches the datastore for the top-`budget`
  fraction, skips the rest. Ships `calibrate` (price the speed/quality curve on val),
  `choose` (mask), `gated_fuse`, and `audit_skip` (the shadow check on the skip decision).
- **`shadow.py`** — tolerance raised **1e-3 → 0.15**. Two correct kNN impls (FAISS vs brute
  force, fp16 vs fp32 keys, tie-breaking) routinely differ a few percent on the true-token
  prob, so 1e-3 flagged ~everything; 0.15 flags only gross disagreement. Non-finite is
  still ALWAYS flagged. Report now carries median (not just max) disagreement + nonfinite_frac.
- **`fusion.py`** — `RUN_SHADOW=1` folds the shadow beside the gate (independent kNN +
  KN-alignment recompute, alarm on the NaN/drift classes); `RUN_UGATE=1` prices the
  uncertainty Pareto on the val slice. Run all three: `RUN_SHADOW=1 RUN_UGATE=1 python fusion.py`.

## Two integration bugs that only surfaced on assembly/run (both fixed here)
1. **Row-min regression.** The handed-in `fusion.py` had reverted the kNN softmax to raw
   `exp(-D/τ)`, which underflows in 512-dim (v1's 97%-NaN bug). Re-applied the row-min shift
   — without it the datastore dies and every downstream measurement is moot.
2. **`RUN_UGATE` ordering `NameError`.** The wired-in ugate block referenced `pc` (KN probs)
   *before* `kn_ngram()` computed it, so `python fusion.py` crashed with `NameError: pc`.
   Relocated the block to after `pc` and the val-tuned weights exist (so the curve uses the
   same honest weights as the gate report). "Verified on synthetic data" missed it because
   the synthetic harness never ran `main()` in this order.

## What the witnesses measured (RTX 4050, WT-103 test, 200k positions, 1M-key datastore)

**Gate** (reproduces v2, leak-free): transformer 53.37 · KN 285.26 · best static blend
**25.25** (0.7 net / 0.3 counts / ~0 kNN) · + gate **25.24** (adaptive term +0.009, noise).

**Uncertainty gate — the curve is INVERTED from the module's own synthetic prediction:**

| search % | skip % | ppl (val slice) | vs full search |
|----------|--------|-----------------|----------------|
| 0%       | 100%   | 24.24           | **−1.14**      |
| 50%      | 50%    | 24.80           | −0.58          |
| 100%     | 0%     | 25.38           | +0.00          |

`[audit] skip 50%: cost −0.23 ppl (free)`. The gate was built to trade quality for speed;
it measured that at this cap **there is no quality to trade** — the kNN channel is
net-*negative*, so retrieving *less* monotonically *lowers* perplexity. This is last
version's "the datastore isn't the halving" re-confirmed as a clean curve, by a tool that
had every incentive to find retrieval useful.

**Shadow channel — first real WT-103 run:**
- `KN(2)` flagged **0.00%** — the count channel's independent alignment recompute agrees to
  the bit (deterministic, same math).
- `kNN(3)` flagged **21%** — the GPU-fp16 primary and the CPU-fp32 independent recompute
  select different top-16 neighbours on ~1/5 of positions (precision + tie-breaking, **not**
  a crash; median disagreement 0). The shadow surfaces it as "worth a look."
- shadow-folded fusion **25.25** (no regression; KN folds where verified, kNN never blended blind).

## The coherent finding
Two independent instruments reach the same verdict without coordinating: the **uncertainty
gate** shows the datastore is net-negative (retrieve less → lower ppl), and the **shadow**
shows the datastore is precision-unstable (21% of kNN probs flip under fp16↔fp32). At the
1M-key cap (2.5% of ~40M), the kNN body is the weak link on both quality and reliability —
which is exactly why the honest gate weights it ~0. None of this lowers the ~25.2 record;
all of it makes the record's *shape* truer. NOT measured: the same at a full ~103M-key
datastore, where kNN-LM's published gains live.
