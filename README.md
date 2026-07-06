# WikiText-103 Fusion Kit — the fair fight, on a GPU

This is the whole ML wing, rebuilt to run on a GPU at real scale, so the numbers
can finally stand next to the "big boys" (Kneser-Ney ~48, GPT-2 ~20 on WikiText-103
word-level perplexity). Everything here is what was prototyped on a 1-core / 4 GB
CPU sandbox — where the transformer half was infeasible. On your GPU it isn't.

The three bodies of the fusion, each earning a comparable number:

  1. counts   — a real **modified/interpolated Kneser-Ney 5-gram** (the ~48 baseline)
  2. neural   — a **transformer** trained from scratch on WikiText-103 (the ~20 target)
  3. kNN      — a **datastore** of the transformer's hidden states (semi-parametric / kNN-LM)

...then fused with a **retrieval-reliability gate** (weight the kNN body by how close the
nearest neighbour is). ⚠ Measured honestly, that gate buys ≈0 over a properly-searched
static blend — see "Measured on a laptop" below; it is kept as honest plumbing, not sold
as the win it was first written up to be.

--------------------------------------------------------------------------------
## Requirements
- Python 3.9+, a CUDA GPU (anything from a laptop 3060 up; more VRAM → bigger model).
- `pip install torch --index-url https://download.pytorch.org/whl/cu121`  (match your CUDA)
- `pip install numpy pyarrow scikit-learn tqdm`
- Optional but recommended for the datastore at scale: `pip install faiss-gpu` (or faiss-cpu)

## Run order
```
python prepare_data.py        # downloads WikiText-103, builds vocab, tokenizes  (~5 min, CPU)
python kn_baseline.py         # modified-KN 5-gram — the count body            (~10 min, CPU, RAM-heavy)
python train.py               # trains the transformer                          (hours, GPU)
python build_datastore.py     # encodes train into the kNN datastore            (~30 min, GPU)
python fusion.py              # counts + neural + kNN + gate, on test            (~15 min, GPU)
```
All scripts read `config.py`. Edit the model size / vocab there.

**The witnesses** (optional, compose into `fusion.py` via env flags):
```
RUN_SHADOW=1 python fusion.py         # shadow.py — independent recompute + fold-where-verified
RUN_UGATE=1  python fusion.py         # uncertainty_gate.py — price retrieve-only-where-unsure
python verify.py                      # offline proof the shadow catches NaN + drift (no GPU)
python model_id.py wt103_data/transformer.pt   # checkpoint fingerprint (arch/weights/state)
```

--------------------------------------------------------------------------------
## What to expect (word-level test perplexity, WikiText-103)

| body / method                         | this kit (default cfg) | field reference |
|---------------------------------------|------------------------|-----------------|
| interpolated Kneser-Ney 5-gram        | ~45–55                 | ~48 (canonical) |
| transformer (d=512, 8L, ~30M params)  | ~28–35                 | GPT-2 sm ~29    |
| transformer (d=768, 12L, ~120M)       | ~20–24                 | GPT-2 lg ~18–20 |
| + kNN-LM fusion                        | −2 to −4 off the net   | kNN-LM: −2 to −3|
| + reliability gate (adaptive kNN wt)  | ≈0 measured (+0.009)†  | (our addition)  |

The point: with a full vocab and a real transformer, these are **directly comparable**
to published numbers — no UNK inflation, standard corpus, canonical splits. The count
ring will land right on ~48 (it's the same algorithm); the transformer lands where its
size puts it on the scaling curve; the fusion shaves the kNN-LM margin off the top.

## Notes / knobs
- **Vocabulary.** `config.VOCAB_SIZE` default 50k (word-level, low UNK). Set to `None`
  to use the full ~267k vocab for the strictly-canonical comparison (bigger embedding).
  BPE/subword would remove UNK entirely — swap the tokenizer in `prepare_data.py` if you
  want to match modern setups exactly.
- **Model size** is the main lever (Chinchilla: ~20 tokens/param is compute-optimal;
  WT103 is ~100M tokens, so ~5M-param is "optimal" but bigger trains fine and wins on
  this data). `config.py` presets: `small` (30M), `medium` (120M).
- **Mixed precision** (`torch.cuda.amp`) is on by default — halves memory, ~2× speed.
- **The domain-routing / cardinal-hue layer is NOT here** — WikiText-103 is unlabelled,
  so there are no domains to route. That machinery (hue conditioning, mode×domain silo,
  soft routing) belongs to the *labelled* 8-domain corpus; to scale that, add the domain
  embeddings back into `train.py`'s model and route the count cells softly (the routing
  race showed: soft/dense, never hard top-1). This kit is the neutral, comparable-to-field
  substrate; your architecture rides on top of it.

## Honesty
Perplexity is only comparable on an identical setup (corpus, vocab, tokenization). This
kit fixes all three to the WikiText-103 standard, which is the whole reason it earns the
right to be compared. The CPU-sandbox numbers from the arc (71, 82.7, etc.) never did —
5k vocab with heavy UNK measured an easier, non-comparable task. This is the fix.

— built from the arc; the wall was always tokens and FLOPs, not ideas.

--------------------------------------------------------------------------------
## Measured on a laptop (July 2026, RTX 4050 6GB) — actually run, not estimated

† The gate row above says ≈0 because that is what it **measured**, leak-free. The story below
is what running the kit — instead of describing it — actually found. Each step corrected the
step before it.

- **KN bug found and fixed.** The first cut of `kn_baseline.py` overflowed int64 for a 50k
  vocab at order >= 4 and silently hashed the *stored* keys while the query kept the exact
  base-V key — so every 4/5-gram lookup missed and the "5-gram" scored **2138** test ppl,
  *worse than a bigram*. `kn_baseline.py` now hashes both sides consistently. Fixed, the
  perplexity decreases monotonically with order (12M-token subset, 50k vocab):
  order-2 **408** → order-3 **307** → order-4 **286** → order-5 **281**. The aspirational
  ~48 needs the full 83M run + full 3-discount modified KN — **it was a target, not a
  reproduced number.**

- **Transformer.** The `small` preset is **76,552,192 params** by `model_id.py`'s fingerprint
  — the untied 50k embeddings make it far bigger than the "~30M" note *and* bigger than the
  "51M" an earlier version of this file guessed. It trains at ~2.8GB VRAM, micro-batch 8,
  AMP, ~96% util; converged to val ppl ~51 (test 52.45) at ~58k steps.

- **The fusion, measured leak-free (v2).** Weights tuned on a val slice, reported on a
  held-out slice (an earlier run tuned on the slice it reported — a leak). Held-out:
  transformer 53.37 · KN 285.26 · **best static blend 25.25** (0.7 net / 0.3 KN / ~0 kNN) ·
  **+ gate 25.24**. The adaptive gate buys **+0.009 ppl** — within noise. The earlier
  "−1.21 gate win" compared a hand-set static blend against a test-slice-tuned gate; it did
  not survive an honest search. **The datastore is not the halving** either: the honest
  search weights kNN ≈0, so the 53→25 drop is the transformer×KN mixture. (kNN-LM's −2/−4
  needs the full ~103M-key store; this cap is 1M.)

- **Two witnesses, one verdict (v3).** `RUN_UGATE=1` prices retrieving only where the LM is
  unsure — and found the curve **inverted**: 0% search = 24.24 ppl, 100% search = 25.38
  (skipping 50% *saves* 0.23 ppl). At this cap retrieving *less* is strictly better, because
  the kNN body is net-negative. `RUN_SHADOW=1` independently recomputes the aux bodies: the
  KN channel agrees to the bit (0% flagged), but the kNN channel diverges on **21%** of
  positions between GPU-fp16 and CPU-fp32 (precision/tie-break, not a crash). Net-negative
  *and* precision-unstable — the datastore is the weak link on both counts, which is why the
  honest gate weights it ~0.

- **Identity is a fact now.** `model_id.py` writes a per-checkpoint fingerprint (sha256 of
  arch-signature / tensor bytes / training state); `diff` two of them to answer "same model?"
  without trusting anyone's memory. It's what corrected the param count above.

**Nothing here lowered the ~25.2 record. Every step made its *shape* truer** — which, for a
kit whose whole pitch is comparability, is the point.
