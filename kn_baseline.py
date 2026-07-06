"""Interpolated Kneser-Ney n-gram (default 5-gram) on WikiText-103 — the count body.

Full-vocab, canonical corpus/split => directly comparable to the published ~48 KN baseline.
Vectorised counting via numpy; RAM-heavy at 5-gram over 100M tokens (expect ~10-30 GB) —
lower ORDER or subsample NTRAIN if memory-bound. Uses interpolated KN with per-order
absolute discounts D = n1/(n1+2*n2) and continuation counts (Chen & Goodman).
"""
import os, math, numpy as np, pickle
import config as C

ORDER  = 5
NTRAIN = None          # None = all; set an int to cap tokens if RAM-bound

def kn_ngram(order=ORDER):
    vocab = pickle.load(open(os.path.join(C.DATA_DIR, "vocab.pkl"), "rb"))
    V = vocab["unk"] + 1
    tr = np.load(os.path.join(C.DATA_DIR, "train_ids.npy")).astype(np.int64)
    te = np.load(os.path.join(C.DATA_DIR, "test_ids.npy")).astype(np.int64)
    if NTRAIN: tr = tr[:NTRAIN]
    print(f"KN-{order}gram | train {len(tr):,} | vocab {V}")

    # encode k-grams as mixed-radix ints (needs V**order < 2**63; ok for V<=~50k at order<=5 via python bigint→object? no)
    # to stay in int64 we hash high-order grams; low collision at these sizes.
    def gram_keys(arr, k):
        if V ** k < 2**62:
            key = arr[:len(arr)-k+1].copy()
            for j in range(1, k):
                key = key * V + arr[j:len(arr)-k+1+j]
            return key
        # fallback: 64-bit polynomial hash (rare collisions, negligible on perplexity)
        h = np.zeros(len(arr)-k+1, dtype=np.uint64)
        for j in range(k):
            h = h * np.uint64(1000003) + arr[j:len(arr)-k+1+j].astype(np.uint64)
        return h.astype(np.int64)

    # highest order counts + context counts + follow(N1+) via sorted grouping
    hi = gram_keys(tr, order)
    hk, hc = np.unique(hi, return_counts=True)
    # KN needs, at each order m: continuation counts for m<order, raw counts for m=order.
    # We build a light chain: for each order, (context, follow, cont) tables via reduceat.
    def tables(keys, counts, radix):
        # keys sorted; split off last symbol => context = keys//radix, sym = keys%radix
        ctx = keys // radix
        starts = np.concatenate([[0], np.where(np.diff(ctx) != 0)[0] + 1])
        ctx_u = ctx[starts]
        ctx_sum = np.add.reduceat(counts, starts).astype(np.float64)          # sum over sym
        follow  = np.diff(np.concatenate([starts, [len(ctx)]])).astype(np.float64)  # distinct sym
        return ctx_u, ctx_sum, follow

    # continuation counts: cont(k-1 gram w) = # distinct preceding words => group hi by suffix
    def continuation(keys, k, radix):
        # suffix = last (k-1) symbols = keys % radix**(k-1)
        mod = radix ** (k - 1)
        suf = np.sort(keys % mod)
        starts = np.concatenate([[0], np.where(np.diff(suf) != 0)[0] + 1])
        suf_u = suf[starts]
        cont  = np.diff(np.concatenate([starts, [len(suf)]])).astype(np.float64)   # distinct predecessors
        return suf_u, cont

    def disc(counts):
        n1 = float((counts == 1).sum()); n2 = float((counts == 2).sum())
        return n1 / (n1 + 2 * n2 + 1e-9)

    # Precompute per-order structures top-down.
    levels = []   # list of dicts per order m = order..1
    cur_keys, cur_cnt = hk, hc.astype(np.float64)
    for m in range(order, 1, -1):
        radix = V
        ctx_u, ctx_sum, follow = tables(cur_keys, cur_cnt, radix)
        d = disc(cur_cnt if m == order else cur_cnt)     # discount from this level's counts
        levels.append(dict(m=m, keys=cur_keys, cnt=cur_cnt, ctx_u=ctx_u, ctx_sum=ctx_sum,
                           follow=follow, d=d))
        # lower order uses continuation counts of the (m-1)-grams
        suf_u, cont = continuation(cur_keys, m, radix)
        cur_keys, cur_cnt = suf_u, cont
    # unigram continuation
    uni = np.zeros(V); uni[cur_keys % V] += cur_cnt if len(cur_cnt) else 0
    # simpler robust unigram continuation: distinct predecessors per word from bigram set
    bik = np.unique(tr[:-1] * V + tr[1:])
    cont_c = np.bincount(bik % V, minlength=V).astype(np.float64)
    Nbi = float(len(bik)); d1 = 0.75

    def look(keys, vals, q):
        i = np.clip(np.searchsorted(keys, q), 0, len(keys) - 1)
        return np.where(keys[i] == q, vals[i], 0.0)

    # ---- evaluate on test via the KN recursion ----
    def kn_prob(positions):
        n = len(positions[0])
        # unigram continuation prob
        c = positions[-1]
        p = np.maximum(cont_c[c] - d1, 0) / Nbi + (d1 * (cont_c > 0).sum() / Nbi) * (1.0 / V)
        # climb orders 2..order
        for lev in reversed(levels):           # from low order up to high
            m = lev["m"]; radix = V
            # context = last m-1 tokens, gram = last m tokens
            ctxq = np.zeros(n, dtype=np.int64); gramq = np.zeros(n, dtype=np.int64)
            for j in range(m):
                gramq = gramq * radix + positions[-m + j]
            for j in range(m - 1):
                ctxq = ctxq * radix + positions[-m + j]
            gc = look(lev["keys"], lev["cnt"], gramq)
            cs = look(lev["ctx_u"], lev["ctx_sum"], ctxq)
            fo = look(lev["ctx_u"], lev["follow"], ctxq)
            d = lev["d"]
            has = cs > 0
            lam = np.where(has, d * fo / np.maximum(cs, 1e-9), 1.0)
            hp = np.where(has, np.maximum(gc - d, 0) / np.maximum(cs, 1e-9), 0.0)
            p = hp + lam * p
        return p

    # build position columns for the test set (need `order` history)
    cols = [te[i:len(te) - order + 1 + i] for i in range(order)]
    p = kn_prob(cols)
    ppl = math.exp(np.mean(-np.log(np.clip(p, 1e-12, None))))
    print(f"\nKneser-Ney {order}-gram  test perplexity: {ppl:.2f}")
    print("(full-vocab canonical setup => compare directly to the published ~48)")
    return ppl, cols, p

if __name__ == "__main__":
    kn_ngram()
