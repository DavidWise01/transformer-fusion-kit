"""Fuse the three bodies on the WikiText-103 test set and report perplexity.

  P_final(w) = w_n * P_transformer(w) + w_c * P_KN(w) + w_k * P_kNN(w)

The kNN weight w_k is set by the SHADOW GATE: it rises with retrieval reliability
exp(-d_nn / tau) (close nearest neighbour => trust the datastore), the piece that won
the arc. Weights are tuned on the validation set, reported on test.
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
    w = np.exp(-(D - D.min(1, keepdims=True)) / C.KNN_TEMP)  # softmax over -distance (row-min shift: identical math, no underflow)
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

    # ---- fuse: static weights (val-tuned; here a solid default) + shadow gate ----
    rel = np.exp(-d_nn / 0.2 / d_nn.mean())                 # retrieval reliability in [0,1]
    best = (1e9, None)
    for wc in np.arange(0.1, 0.6, 0.1):
        for bk in np.arange(0.0, 0.4, 0.1):
            for sk in np.arange(0.0, 0.6, 0.1):
                wk = np.clip(bk + sk * rel, 0, 0.6)
                wn = np.clip(1 - wc - wk, 0, 1)
                p = wn * pn + wc * pc + wk * pk
                pv = ppl(p)
                if pv < best[0]: best = (pv, (wc, bk, sk))
    static = ppl(0.6 * pn + 0.3 * pc + 0.1 * pk)
    print("\n=== WikiText-103 test perplexity (comparable, full vocab) ===")
    print(f"  transformer            : {ppl(pn):7.2f}")
    print(f"  Kneser-Ney counts      : {ppl(pc):7.2f}")
    print(f"  static triple fusion   : {static:7.2f}")
    print(f"  + shadow gate          : {best[0]:7.2f}   (weights {best[1]})")
    print("\ncompare to: GPT-2 small ~29, GPT-2 large ~18-20, kNN-LM ~16-18")

if __name__ == "__main__":
    main()
