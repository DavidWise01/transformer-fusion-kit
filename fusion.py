"""Fuse the three bodies on the WikiText-103 test set and report perplexity.

  P_final(w) = w_n * P_transformer(w) + w_c * P_KN(w) + w_k * P_kNN(w)

The kNN weight w_k is set by the reliability gate: it rises with retrieval reliability
exp(-d_nn / tau) (close nearest neighbour => trust the datastore). Weights are tuned on
a validation slice and reported on a held-out slice — never on the slice being scored.
Two optional witnesses compose in (env flags, default on):
  RUN_SHADOW=1  the shadow CHANNEL — independent recompute of the aux bodies, folds
                2->1 only where verified, alarms on the NaN-underflow / drift classes.
  RUN_UGATE=1   the uncertainty gate — prices the speed/quality curve of retrieving
                only where the LM is unsure (a dial, not a free lunch).
"""
import os, math, numpy as np, pickle, torch, torch.nn.functional as F
import config as C
from model import build_model
from kn_baseline import kn_ngram

def ppl(p, m=None):
    v = p if m is None else p[m]
    return math.exp(np.mean(-np.log(np.clip(v, 1e-12, None))))

@torch.no_grad()
def neural_and_keys(model, ids, ctx, device, cap):
    """Return, for up to `cap` test positions: P_transformer(true), hidden key, true token."""
    pn, keys, tru = [], [], []
    got = 0
    for start in range(0, len(ids) - ctx - 1, ctx):
        x = torch.from_numpy(ids[start:start+ctx].astype(np.int64))[None].to(device)
        nxt = ids[start+1:start+1+ctx]
        with torch.autocast("cuda", enabled=C.AMP):
            logits, hid = model(x, return_hidden=True)
            probs = F.softmax(logits.float(), -1)[0]     # [ctx, V]
        for t in range(ctx):
            pn.append(float(probs[t, nxt[t]])); keys.append(hid[0, t].float().cpu().numpy()); tru.append(int(nxt[t]))
            got += 1
            if got >= cap: break
        if got >= cap: break
    return np.array(pn), np.array(keys, dtype=np.float32), np.array(tru)

def knn_probs(keys, tru, device):
    """kNN-LM distribution over the true token + nearest-neighbour distance, via FAISS or brute force."""
    dk = np.load(C.DS_PATH + ".keys.npy"); dv = np.load(C.DS_PATH + ".vals.npy")
    try:
        import faiss
        index = faiss.read_index(C.DS_PATH + ".faiss")
        if device == "cuda": index = faiss.index_cpu_to_all_gpus(index)
        D, I = index.search(keys.astype(np.float32), C.KNN_K)
        nn_vals = dv[I]
    except Exception:
        dk_t = torch.from_numpy(dk.astype(np.float32)).to(device)
        dn = (dk_t ** 2).sum(1)
        D_list, V_list = [], []
        for i in range(0, len(keys), 256):
            q = torch.from_numpy(keys[i:i+256]).to(device)
            dist = (q ** 2).sum(1, keepdim=True) + dn[None] - 2 * q @ dk_t.T
            dd, ii = torch.topk(dist, C.KNN_K, dim=1, largest=False)
            D_list.append(dd.cpu().numpy()); V_list.append(dv[ii.cpu().numpy()])
        D = np.concatenate(D_list); nn_vals = np.concatenate(V_list)
    D = np.clip(D, 0, None)
    w = np.exp(-(D - D.min(1, keepdims=True)) / C.KNN_TEMP)  # softmax over -distance (row-min shift: no underflow)
    w = w / w.sum(1, keepdims=True)
    pk = (w * (nn_vals == tru[:, None])).sum(1)            # prob mass on the true token
    return pk, D[:, 0]                                     # + nearest-neighbour distance

def kn_probs_for(cols_positions_true, kn_ppl_cols):
    # kn_ngram returns (ppl, cols, p) over the *test* trigram positions; align by true token index.
    return kn_ppl_cols

def main():
    vocab = pickle.load(open(os.path.join(C.DATA_DIR, "vocab.pkl"), "rb"))
    Vsize = vocab["unk"] + 1
    te = np.load(os.path.join(C.DATA_DIR, "test_ids.npy"))
    ctx = C.MODEL["ctx"]
    model = build_model(Vsize)
    model.load_state_dict(torch.load(C.CKPT, map_location=C.DEVICE)["model"]); model.eval()

    CAP = min(len(te) - ctx - 1, 200000)
    print("evaluating transformer + collecting keys...")
    pn, keys, tru = neural_and_keys(model, te, ctx, C.DEVICE, CAP)
    print(f"  transformer perplexity: {ppl(pn):.2f}")

    print("kNN search over datastore...")
    pk, d_nn = knn_probs(keys, tru, C.DEVICE)
    print(f"  neighbour recall: {100*np.mean(pk>0):.1f}%")

    print("Kneser-Ney counts...")
    kn_ppl, kn_cols, kn_p = kn_ngram()                     # p over test positions (order-offset)
    # align KN (starts at position `order-1`) with our per-token arrays (start at 0)
    off = len(te) - len(kn_p)                               # order-1
    pc = np.ones_like(pn)
    m = min(len(pc) - off, len(kn_p))
    pc[off:off+m] = kn_p[:m]

    # ---- input integrity guard (was silently defeating the grid search) ----
    # A NaN anywhere in a channel makes `pv < best` always False, so a poisoned
    # run reported "weights None" and the fault was never seen. Refuse to fuse
    # blind: report which channel is corrupt and clean it to a safe floor.
    for name, arr in (("transformer", pn), ("KN", pc), ("kNN", pk), ("d_nn", d_nn)):
        bad = ~np.isfinite(arr)
        if bad.any():
            print(f"  [GUARD] {name}: {bad.mean():.2%} non-finite entries "
                  f"-> clamped to floor; investigate before trusting numbers")
    pn = np.nan_to_num(pn, nan=1e-12); pc = np.nan_to_num(pc, nan=1e-12)
    pk = np.nan_to_num(pk, nan=0.0)
    d_nn = np.nan_to_num(d_nn, nan=np.nanmax(d_nn[np.isfinite(d_nn)]) if np.isfinite(d_nn).any() else 1.0)

    # ---- retrieval-reliability gate (kept as a gate; plumbing fixed) ----
    # Old line `np.exp(-d_nn/0.2/d_nn.mean())` saturated to ~0 at the mean distance
    # (exp(-5)); rel did almost nothing and the win came from the static term.
    # Scale by the distance spread so `rel` spans a usable range, and TUNE ON VAL,
    # never on the test slice being reported.
    scale = d_nn.std() + 1e-9
    rel = np.exp(-(d_nn - d_nn.min()) / scale)             # in (0,1], 1 = closest nbr
    rel = (rel - rel.min()) / (rel.max() - rel.min() + 1e-9)

    # split a validation slice off the FRONT; tune weights there, report on the rest
    nval = len(pn) // 5
    val = slice(0, nval); tst = slice(nval, None)

    def fuse(wc, bk, sk, sl):
        wk = np.clip(bk + sk * rel[sl], 0, 0.9)
        wn = np.clip(1 - wc - wk, 0, 1)
        return wn * pn[sl] + wc * pc[sl] + wk * pk[sl]

    def safe_ppl(p):
        v = ppl(p)
        return v if np.isfinite(v) else np.inf          # explicit: inf never "wins"

    # search static-only (sk=0) and full-gate; compare on val, report best on test
    best_static = (np.inf, None); best_gate = (np.inf, None)
    for wc in np.arange(0.1, 0.6, 0.1):
        for bk in np.arange(0.0, 0.4, 0.1):
            v0 = safe_ppl(fuse(wc, bk, 0.0, val))          # no rel term
            if v0 < best_static[0]: best_static = (v0, (wc, bk, 0.0))
            for sk in np.arange(0.1, 0.6, 0.1):
                vg = safe_ppl(fuse(wc, bk, sk, val))       # with rel term
                if vg < best_gate[0]: best_gate = (vg, (wc, bk, sk))

    # report on the held-out test slice with the val-selected weights
    static_te = safe_ppl(fuse(*best_static[1], tst))
    gate_te   = safe_ppl(fuse(*best_gate[1], tst))
    rel_gain  = static_te - gate_te                        # what the adaptive rel term actually buys

    print("\n=== WikiText-103 test perplexity (val-tuned, reported on held-out slice) ===")
    print(f"  transformer            : {ppl(pn[tst]):7.2f}")
    print(f"  Kneser-Ney counts      : {ppl(pc[tst]):7.2f}")
    print(f"  best static blend      : {static_te:7.2f}   (weights {best_static[1][:2]})")
    print(f"  + reliability gate      : {gate_te:7.2f}   (weights {best_gate[1]})")
    print(f"  [DIAGNOSTIC] adaptive rel term buys: {rel_gain:+.3f} ppl over the best static blend")
    if best_gate[1][2] == 0.0 or abs(rel_gain) < 0.05:
        print("  [DIAGNOSTIC] -> the gate's win is essentially the static term; rel barely contributes.")

    # ---- uncertainty-gate CALIBRATION (price the speed/quality curve) ----
    # Runs AFTER pc/pk/val exist and the weights are val-tuned, so the curve uses the
    # same honest weights as the gate report. It computes pk for all positions (already
    # done) to MEASURE what skipping would cost; the actual speedup comes later, by NOT
    # searching off-mask inside knn_probs once you pick a budget.
    if os.environ.get("RUN_UGATE", "1") == "1":
        import uncertainty_gate as UG
        w_kn_u = best_static[1][0]                        # the val-tuned KN weight
        w_knn_u = 0.1                                     # small kNN weight to price (static picks ~0)
        unc = UG.uncertainty_from_true(pn[val])          # coarse proxy from true-tok prob
        rows = UG.calibrate(pn[val], pc[val], pk[val], unc,
                            w_kn=w_kn_u, w_knn=w_knn_u, ppl_fn=ppl)
        print("\n=== uncertainty gate: speed/quality Pareto (val slice, priced) ===")
        print(f"  {'search%':>8} {'skip%':>7} {'ppl':>9}")
        full = next(r for r in rows if r['budget'] >= 1.0)['ppl']
        for r in rows:
            print(f"  {r['searched_frac']*100:7.1f}% {r['skipped_frac']*100:6.1f}% "
                  f"{r['ppl']:9.4f}   ({r['ppl']-full:+.4f} vs full search)")
        mask = UG.choose(unc, budget=0.5)
        a = UG.audit_skip(pn[val], pc[val], pk[val], mask, w_kn_u, w_knn_u, ppl, sample=2000)
        print(f"  [audit] skip 50%: cost {a['cost']:+.4f} ppl "
              f"({'free' if a['cost']<0.05 else 'priced — retrieval helps even mid-confidence'})")

    # ---- shadow channel: verify 2/3/4, fold 2 into 1 conditioned on agreement ----
    # Opt-in (RUN_SHADOW=1): the kNN recompute is a second independent datastore
    # search, so it doubles kNN cost. The KN alignment recompute is free.
    if os.environ.get("RUN_SHADOW", "1") == "1":
        import shadow as SH
        print("\nshadow channel: independent recompute + verify + fold(2->1)...")
        # (3) independent kNN recompute from the datastore (separate impl from knn_probs)
        dstore_k = np.load(C.DS_PATH + ".keys.npy")
        dstore_v = np.load(C.DS_PATH + ".vals.npy")
        pk_shadow, _ = SH.recompute_knn(keys, tru, dstore_k, dstore_v,
                                        C.KNN_K, C.KNN_TEMP)
        # (2) independent KN alignment recompute (catches the off-by-`order` drift)
        pc_shadow, kn_align_bad = SH.recompute_kn_alignment(pc, kn_p, off)
        if kn_align_bad:
            print("  [shadow] KN alignment INVALID (off/len mismatch) -> KN fold refused")

        # fold on the SAME held-out slice used above, so it's comparable
        sl = tst
        fused_sh, rep = SH.shadow_fold(
            pn[sl], pc[sl], pk[sl],
            pc_shadow=pc_shadow[sl], pk_shadow=pk_shadow[sl],
            w_kn=best_static[1][0],           # reuse the val-tuned KN weight
            verbose=True,
        )
        print(SH.shadow_report_line(rep))
        print(f"  shadow-folded fusion    : {safe_ppl(fused_sh):7.2f}   "
              f"(2->1 where verified; primary alone where flagged)")
        # cross-check: did primary kNN and shadow kNN agree? (a real value-drift alarm)
        knn_max_dis = rep["kNN(3)"]["max_disagreement"]
        if rep["kNN(3)"]["any_nonfinite"]:
            print("  [shadow] ALARM: primary kNN produced non-finite probs — "
                  "the NaN-underflow class. Investigate knn_probs before trusting fusion.")
        elif knn_max_dis > 0.05:
            print(f"  [shadow] NOTE: primary vs independent kNN disagree "
                  f"(max {knn_max_dis:.2%}) — the two kNN impls diverge; worth a look.")

    print("\ncompare to: GPT-2 small ~29, GPT-2 large ~18-20, kNN-LM ~16-18")

if __name__ == "__main__":
    main()
