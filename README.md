# WikiText-103 Fusion Kit — the fair fight, on a GPU

This is the whole ML wing, rebuilt to run on a GPU at real scale, so the numbers
can finally stand next to the "big boys" (Kneser-Ney ~48, GPT-2 ~20 on WikiText-103
word-level perplexity). Everything here is what was prototyped on a 1-core / 4 GB
CPU sandbox — where the transformer half was infeasible. On your GPU it isn't.

The three bodies of the fusion, each earning a comparable number:

  1. counts   — a real **modified/interpolated Kneser-Ney 5-gram** (the ~48 baseline)
  2. neural   — a **transformer** trained from scratch on WikiText-103 (the ~20 target)
  3. kNN      — a **datastore** of the transformer's hidden states (semi-parametric / kNN-LM)

...then fused with a **retrieval-reliability shadow gate** (weight the kNN body by how
close the nearest neighbour is), the piece that won the arc.

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
python fusion.py              # counts + neural + kNN + shadow gate, on test     (~15 min, GPU)
```
All scripts read `config.py`. Edit the model size / vocab there.

--------------------------------------------------------------------------------
## What to expect (word-level test perplexity, WikiText-103)

| body / method                         | this kit (default cfg) | field reference |
|---------------------------------------|------------------------|-----------------|
| interpolated Kneser-Ney 5-gram        | ~45–55                 | ~48 (canonical) |
| transformer (d=512, 8L, ~30M params)  | ~28–35                 | GPT-2 sm ~29    |
| transformer (d=768, 12L, ~120M)       | ~20–24                 | GPT-2 lg ~18–20 |
| + kNN-LM fusion                        | −2 to −4 off the net   | kNN-LM: −2 to −3|
| + shadow gate (adaptive kNN weight)   | a further small drop   | (our addition)  |

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
