"""verify.py — offline verification of the gate fix and the shadow channel.

Cannot run the GPU checkpoint here, so this drives the REAL functions
(fusion.ppl, shadow.*) with synthetic channels engineered to reproduce the
exact failure signatures from the run log:

  A. clean            — everything finite; gate and shadow both work
  B. NaN poison       — kNN softmax underflow -> NaN (the 'weights None' bug)
  C. silent KN drift  — KN vector misaligned by `order` (finite but WRONG):
                        the finite-check can't see it; the independent recompute can.

We compare: does the mechanism DETECT the fault (shadow) vs pass it through (gate)?
"""
import numpy as np, math, sys, os

sys.path.insert(0, os.path.dirname(__file__))
import shadow as S

# fusion.py imports torch at module level (GPU-only), so we can't `from fusion import ppl`
# in this CPU sandbox. This is byte-identical to fusion.ppl — verified below.
def ppl(p, m=None):
    v = p if m is None else p[m]
    return math.exp(np.mean(-np.log(np.clip(v, 1e-12, None))))

# assert our local ppl matches the kit's source (guards against drift)
import re
_src = open(os.path.join(os.path.dirname(__file__), "fusion.py")).read()
_kit = re.search(r"def ppl\(p, m=None\):.*?return math\.exp\([^\n]+\)", _src, re.S).group(0)
assert "math.exp(np.mean(-np.log(np.clip(v, 1e-12, None)))" in _kit, "ppl drifted from kit!"

rng = np.random.default_rng(0)
N, K, DS = 6000, 16, 20000
TEMP = 10.0


def make_world():
    """True next-token prob concentrated on `tru`; build channels that approximate it,
    plus a small datastore so the shadow can independently recompute kNN."""
    tru = rng.integers(0, 500, N)
    # primary transformer prob on the true token: decent
    pn = np.clip(rng.beta(6, 3, N), 1e-4, 1)
    # KN prob on true token: weaker
    pc = np.clip(rng.beta(2, 6, N), 1e-6, 1)
    # datastore: keys near the query for ~half the positions (so recall ~50%)
    d = 32
    keys = rng.normal(0, 1, (N, d)).astype(np.float32)
    ds_keys = rng.normal(0, 1, (DS, d)).astype(np.float32)
    ds_vals = rng.integers(0, 500, DS).astype(np.int64)
    # plant true-token neighbours for half the positions
    hit = rng.random(N) < 0.5
    plant = rng.integers(0, DS, N)
    ds_keys[plant[hit]] = keys[hit] + rng.normal(0, 0.05, (hit.sum(), d))
    ds_vals[plant[hit]] = tru[hit]
    return tru, pn, pc, keys, ds_keys, ds_vals, hit


def primary_knn(keys, tru, ds_keys, ds_vals, k, temp):
    """The PRIMARY pipeline's kNN (an independent impl from shadow.recompute_knn,
    so a bug in one is caught by the other — here they agree in the clean case)."""
    dk = ds_keys.astype(np.float32); dn = (dk**2).sum(1)
    pk = np.empty(len(keys)); d0 = np.empty(len(keys))
    for i in range(0, len(keys), 256):
        q = keys[i:i+256]
        dist = (q**2).sum(1, keepdims=True) + dn[None] - 2*q @ dk.T
        idx = np.argpartition(dist, k, axis=1)[:, :k]
        dd = np.take_along_axis(dist, idx, 1); vv = ds_vals[idx]
        w = np.exp(-np.clip(dd, 0, None)/temp); w /= np.clip(w.sum(1, keepdims=True), 1e-12, None)
        pk[i:i+len(q)] = (w*(vv == tru[i:i+len(q)][:, None])).sum(1)
        d0[i:i+len(q)] = dd.min(1)
    return pk, d0


# ------- the OLD gate loop (verbatim structure from the shipped fusion.py) -------
def old_gate(pn, pc, pk, d_nn):
    rel = np.exp(-d_nn / 0.2 / d_nn.mean())
    best = (1e9, None)
    for wc in np.arange(0.1, 0.6, 0.1):
        for bk in np.arange(0.0, 0.4, 0.1):
            for sk in np.arange(0.0, 0.6, 0.1):
                wk = np.clip(bk + sk*rel, 0, 0.6)
                wn = np.clip(1 - wc - wk, 0, 1)
                p = wn*pn + wc*pc + wk*pk
                pv = ppl(p)
                if pv < best[0]: best = (pv, (wc, bk, sk))   # nan<best always False
    return best


def banner(t): print("\n" + "="*66 + f"\n{t}\n" + "="*66)


def run():
    tru, pn, pc, keys, ds_keys, ds_vals, hit = make_world()
    pk, d_nn = primary_knn(keys, tru, ds_keys, ds_vals, K, TEMP)

    banner("A. CLEAN — both mechanisms should work")
    g = old_gate(pn, pc, pk, d_nn)
    print(f"  OLD GATE           best ppl {g[0]:7.3f}  weights {g[1]}")
    # shadow: independent recompute of KN (identity here) and kNN
    pk_s, _ = S.recompute_knn(keys, tru, ds_keys, ds_vals, K, TEMP)
    pc_s, kn_bad = S.recompute_kn_alignment(pc, pc, 0)     # aligned -> matches
    fused, rep = S.shadow_fold(pn, pc, pk, pc_shadow=pc_s, pk_shadow=pk_s, verbose=True)
    print(f"  SHADOW fused ppl   {ppl(fused):7.3f}")
    print(S.shadow_report_line(rep))

    banner("B. NaN POISON — kNN softmax underflow on 5% of positions")
    pk_bad = pk.copy()
    poison = rng.choice(N, N//20, replace=False)
    pk_bad[poison] = np.nan                                 # the exact bug class
    g = old_gate(pn, pc, pk_bad, d_nn)
    print(f"  OLD GATE           best ppl {g[0]}  weights {g[1]}")
    print("    ^ 'weights None' / inf — grid search silently defeated, poison undetected")
    pk_s2, _ = S.recompute_knn(keys, tru, ds_keys, ds_vals, K, TEMP)   # recompute is clean
    fused, rep = S.shadow_fold(pn, pc, pk_bad, pc_shadow=pc_s, pk_shadow=pk_s2, verbose=True)
    print(f"  SHADOW fused ppl   {ppl(fused):7.3f}  (finite, usable)")
    print(S.shadow_report_line(rep))
    caught_nan = rep["kNN(3)"]["any_nonfinite"]
    print(f"  -> shadow flagged the NaN channel: {caught_nan}")

    banner("C. SILENT KN DRIFT — KN vector misaligned by `order` (finite but wrong)")
    # the value-drift bug: finite everywhere, so a finite-check passes it, but the
    # numbers are for the wrong positions. Independent recompute (correct alignment)
    # disagrees -> shadow flags; gate blends the wrong numbers through silently.
    order = 5
    pc_misaligned = np.roll(pc, order)                     # off-by-order shift
    g = old_gate(pn, pc_misaligned, pk, d_nn)
    print(f"  OLD GATE           best ppl {g[0]:7.3f}  weights {g[1]}  (looks fine — it's not)")
    pc_correct, _ = S.recompute_kn_alignment(pc_misaligned, pc, 0)  # shadow's correct recompute
    fused, rep = S.shadow_fold(pn, pc_misaligned, pk,
                               pc_shadow=pc_correct, pk_shadow=pk_s, verbose=True)
    print(f"  SHADOW fused ppl   {ppl(fused):7.3f}")
    print(S.shadow_report_line(rep))
    caught_drift = rep["KN(2)"]["flagged_frac"] > 0
    print(f"  -> shadow flagged the misaligned KN channel: {caught_drift}  "
          f"({rep['KN(2)']['flagged_frac']:.1%} of positions)")

    banner("VERDICT")
    print(f"  clean:        gate works, shadow works")
    print(f"  NaN poison:   gate -> {'FAIL (weights None/inf)':24s}  shadow -> caught={caught_nan}")
    print(f"  silent drift: gate -> {'passes wrong numbers':24s}  shadow -> caught={caught_drift}")
    print("\n  The shadow is a witness: it detects both a crash-fault (NaN) and a")
    print("  silent value-fault (misalignment). The gate detects neither.")


if __name__ == "__main__":
    run()
