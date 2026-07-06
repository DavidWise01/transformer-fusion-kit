"""Shared configuration for the WikiText-103 fusion kit."""
import os, torch

# ---- paths ----
DATA_DIR   = os.environ.get("WT_DATA", "./wt103_data")
CKPT       = os.path.join(DATA_DIR, "transformer.pt")
DS_PATH    = os.path.join(DATA_DIR, "datastore")          # .keys.npy / .vals.npy (+ faiss index)
os.makedirs(DATA_DIR, exist_ok=True)

# ---- device ----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP    = (DEVICE == "cuda")                               # mixed precision on GPU

# ---- vocabulary ----
VOCAB_SIZE = 50000        # word-level, top-K; set None for the full ~267k (strictly canonical)

# ---- model size presets ----
PRESET = os.environ.get("WT_PRESET", "small")             # "small" (~30M) or "medium" (~120M)
_MODELS = {
    "small":  dict(d_model=512, n_layers=8,  n_heads=8,  ctx=256),
    "medium": dict(d_model=768, n_layers=12, n_heads=12, ctx=384),
}
MODEL = _MODELS[PRESET]

# ---- training ----
BATCH_TOKENS = 16384          # tokens per optimizer step (grad-accumulate to reach this)
MICRO_BATCH  = 16             # sequences per forward (lower if you OOM)
LR           = 3e-4
WARMUP       = 2000
MAX_STEPS    = 60000          # ~a few epochs on WT103; watch val and stop when it flattens
WEIGHT_DECAY = 0.1
GRAD_CLIP    = 1.0
EVAL_EVERY   = 2000
CKPT_EVERY   = 2000

# ---- kNN / fusion ----
KNN_K        = 16
KNN_TEMP     = 10.0           # softmax temperature over -distance for kNN distribution
DATASTORE_MAX= 40_000_000     # cap datastore keys (RAM/VRAM); None = all train positions
