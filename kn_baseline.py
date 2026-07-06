"""Interpolated Kneser-Ney n-gram (default 5-gram) on WikiText-103 — the count body.

Full-vocab, canonical corpus/split => directly comparable to a published KN baseline.

Every gram, context and suffix is keyed by a CONSISTENT 64-bit hash (FNV-1a style), used
identically at BUILD and QUERY. This is deliberate: packing a gram as a base-V integer
overflows int64 once V**order exceeds ~2**62 (e.g. a 50k vocab at order >= 4), and the
earlier cut then silently hashed the *stored* keys while the query still built the exact
base-V key — so every high-order lookup missed and the model collapsed toward unigrams
(order-5 scored *worse* than a bigram). Hashing both sides removes the overflow and the
mismatch. Uses interpolated KN with a per-order absolute discount D = n1/(n1+2*n2) and
continuation counts (Chen & Goodman). RAM scales with the number of distinct grams;
lower ORDER or subsample NTRAIN if memory-bound.
"""
import os, math, numpy as np, pickle
import config as C

ORDER  = 5
NTRAIN = None          # None = all train tokens; set an int to cap if RAM-bound

# ---- consistent 64-bit hash of a gram given as a list of int columns (first..last) ----
_FNV_PRIME  = np.uint64(1099511628211)
_FNV_OFFSET = np.uint64(1469598103934665603)
_MIX        = np.uint64(0x9E3779B97F4A7C15)
_POS        = np.uint64((1 << 63) - 1)

def _hash_cols(cols):
    """Order-independent, collision-resistant hash. cols[0]=first token .. cols[-1]=last.
    IDENTICAL at build and query — that is the whole fix. Returns non-negative int64."""
    h = np.full(len(cols[0]), _FNV_OFFSET, dtype=np.uint64)
    for c in cols:
        h = (h ^ (c.astype(np.uint64) + _MIX)) * _FNV_PRIME
    return (h & _POS).astype(np.int64)

def _stream_cols(arr, k):
    """k columns over the stream: col j = arr[j : N-k+1+j] (all k-grams, with repetition)."""
    N = len(arr)
    return [arr[j:N - k + 1 + j] for j in range(k)]

def _look(keys, vals, q):
    """vals[i] where keys[i]==q (keys sorted asc); 0.0 where q absent."""
    if len(keys) == 0:
        return np.zeros(len(q), dtype=np.float64)
    i = np.clip(np.searchsorted(keys, q), 0, len(keys) - 1)
    return np.where(keys[i] == q, vals[i], 0.0)

def _uniq_counts(h):
    """sorted unique of h + occurrence counts (float)."""
    u, c = np.unique(h, return_counts=True)
    return u, c.astype(np.float64)

def _sum_by(key, val):
    """sum val grouped by key -> (uniq_key_sorted, summed_float)."""
    o = np.argsort(key, kind="stable")
    k = key[o]; v = val[o].astype(np.float64)
    st = np.concatenate([[0], np.where(np.diff(k) != 0)[0] + 1])
    return k[st], np.add.reduceat(v, st)

def _disc(counts):
    n1 = float((counts == 1).sum()); n2 = float((counts == 2).sum())
    return n1 / (n1 + 2 * n2 + 1e-9)


def kn_ngram(order=ORDER):
    vocab = pickle.load(open(os.path.join(C.DATA_DIR, "vocab.pkl"), "rb"))
    V = vocab["unk"] + 1
    tr = np.load(os.path.join(C.DATA_DIR, "train_ids.npy")).astype(np.int64)
    te = np.load(os.path.join(C.DATA_DIR, "test_ids.npy")).astype(np.int64)
    if NTRAIN:
        tr = tr[:NTRAIN]
    n = order
    print(f"KN-{n}gram | train {len(tr):,} | vocab {V}")

    GRAM, CTX, Dm = {}, {}, {}
    P1 = np.zeros(V, dtype=np.float64)

    # ---- highest order n: RAW counts ----
    cols = _stream_cols(tr, n)
    gram_h = _hash_cols(cols)
    ctx_h  = _hash_cols(cols[:n - 1])
    ug, gcount = _uniq_counts(gram_h)
    Dm[n] = _disc(gcount)
    GRAM[n] = (ug, gcount)
    uctx, csum = _uniq_counts(ctx_h)                       # c(h)
    _, first = np.unique(gram_h, return_index=True)        # one position per unique gram
    fctx, follow = _uniq_counts(ctx_h[first])              # distinct grams per context = N1+(h.)
    CTX[n] = (uctx, csum, _look(fctx, follow, uctx))
    hi_cols = [c[first] for c in cols]                     # unique n-gram token columns

    # ---- lower orders m = n-1 .. 1: CONTINUATION counts ----
    for m in range(n - 1, 0, -1):
        suf_cols = hi_cols[1:]                             # last m tokens of each (m+1)-gram
        suf_h = _hash_cols(suf_cols)                       # == hash of the m-gram (consistent)
        um, contcount = _uniq_counts(suf_h)                # N1+(.x): distinct predecessors of x
        Dm[m] = _disc(contcount)
        GRAM[m] = (um, contcount)
        _, sfirst = np.unique(suf_h, return_index=True)
        m_cols = [c[sfirst] for c in suf_cols]             # unique m-gram columns (in um order)
        if m >= 2:
            mctx_h = _hash_cols(m_cols[:m - 1])            # m-gram context = first m-1
            uc, cs = _sum_by(mctx_h, contcount)            # N1+(.h.) = sum contcount over words
            fc, fo = _uniq_counts(mctx_h)                  # N1+(h.) = distinct m-grams per context
            CTX[m] = (uc, cs, _look(fc, fo, uc))
        else:                                              # m == 1: unigram continuation base
            total = float(contcount.sum())
            P1[m_cols[0]] = contcount / max(total, 1e-9)
        hi_cols = m_cols

    # ---- evaluate on test via the interpolated KN recursion (unigram -> order n) ----
    qcols = _stream_cols(te, n)
    w = qcols[-1]
    p = np.where(P1[w] > 0, P1[w], 1.0 / V)                # unigram continuation (floor unseen)
    for m in range(2, n + 1):
        gram_hash = _hash_cols(qcols[n - m:])
        ctx_hash  = _hash_cols(qcols[n - m:n - 1])
        gc = _look(GRAM[m][0], GRAM[m][1], gram_hash)
        cs = _look(CTX[m][0],  CTX[m][1], ctx_hash)
        fo = _look(CTX[m][0],  CTX[m][2], ctx_hash)
        D  = Dm[m]
        has = cs > 0
        lam = np.where(has, D * fo / np.maximum(cs, 1e-9), 1.0)
        hp  = np.where(has, np.maximum(gc - D, 0.0) / np.maximum(cs, 1e-9), 0.0)
        p = hp + lam * p

    ppl = math.exp(np.mean(-np.log(np.clip(p, 1e-12, None))))
    print(f"\nKneser-Ney {n}-gram  test perplexity: {ppl:.2f}")
    print("(full-vocab canonical setup => compare directly to a published KN baseline)")
    return ppl, qcols, p


if __name__ == "__main__":
    kn_ngram()
