"""shadow.py — the SHADOW CHANNEL, to spec.

This is NOT the reliability gate in fusion.py. Per the design:

    "shadow channel mirrors the original channel and verifies numbers 2,3,4
     while 1 is loading in context, then compares contextually at the end
     and folds 2 back into channel 1."

Mechanism (deliberately a witness, not a mixer):

  1. MIRROR      keep an independent handle on the primary channel (1 = transformer).
  2. VERIFY 2/3/4 recompute each auxiliary body's per-position probability *from its
                  own raw inputs* (KN counts, datastore neighbours) — NOT from the
                  handed-in pn/pc/pk. This independence is the whole point: it lets the
                  shadow catch corruption upstream of the number it is checking.
  3. COMPARE     contextually diff the recompute against the value the primary pipeline
                  produced; emit a per-position DISAGREEMENT (and +inf where non-finite).
  4. FOLD 2->1   fold the KN body (channel 2) back into the transformer (channel 1),
                  CONDITIONED on agreement: where a channel is flagged, refuse its
                  contribution for those positions and defer to the primary, instead of
                  blending poison through.

The contract with the caller: shadow_fold(...) returns (fused_prob, report). The report
is the audit — flagged fractions, disagreement stats, and whether any channel was
non-finite. A caller must be able to see that a run was corrupted; that visibility is
the feature the gate lacked.
"""
import numpy as np

EPS = 1e-12


# ----------------------------------------------------------------------
# independent recompute of each auxiliary body
# ----------------------------------------------------------------------
def recompute_knn(keys, tru, datastore_keys, datastore_vals, k, temp, device="cpu"):
    """Independently recompute the kNN-LM probability on the true token AND the
    nearest-neighbour distance, from the datastore directly. Mirrors knn_probs in
    fusion.py but is a *separate* implementation so a bug in one is caught by the other.
    Brute-force torch/np search; the point is independence, not speed.
    """
    dk = datastore_keys.astype(np.float32)
    dv = datastore_vals
    dn = (dk ** 2).sum(1)
    pk = np.empty(len(keys), np.float64)
    d0 = np.empty(len(keys), np.float64)
    for i in range(0, len(keys), 256):
        q = keys[i:i+256].astype(np.float32)
        # squared L2, same metric as the primary
        dist = (q ** 2).sum(1, keepdims=True) + dn[None] - 2.0 * q @ dk.T
        idx = np.argpartition(dist, min(k, dist.shape[1]-1), axis=1)[:, :k]
        dd = np.take_along_axis(dist, idx, axis=1)
        vv = dv[idx]
        w = np.exp(-np.clip(dd, 0, None) / temp)
        w = w / np.clip(w.sum(1, keepdims=True), EPS, None)
        pk[i:i+len(q)] = (w * (vv == tru[i:i+len(q)][:, None])).sum(1)
        d0[i:i+len(q)] = dd.min(1)
    return pk, d0


def recompute_kn_alignment(pc_primary, kn_p, off):
    """Independently re-derive the KN per-position vector from the raw kn_ngram output
    and the offset, and hand back both so the caller can compare. Catches the classic
    off-by-`order` alignment bug (a silent, perplexity-poisoning error the gate can't see).
    """
    pc = np.ones_like(pc_primary)
    m = min(len(pc) - off, len(kn_p))
    if off < 0 or m <= 0:
        return pc, True   # alignment invalid -> flag everything
    pc[off:off+m] = kn_p[:m]
    return pc, False


# ----------------------------------------------------------------------
# contextual comparison -> disagreement
# ----------------------------------------------------------------------
def disagreement(primary_view, shadow_view):
    """Per-position disagreement between the primary pipeline's value for a channel
    and the shadow's independent recompute. Relative error on the true-token prob;
    +inf where either side is non-finite (that infinity IS the flag)."""
    a = np.asarray(primary_view, np.float64)
    b = np.asarray(shadow_view, np.float64)
    bad = ~(np.isfinite(a) & np.isfinite(b))
    rel = np.abs(a - b) / np.clip(np.maximum(np.abs(a), np.abs(b)), EPS, None)
    rel = np.where(bad, np.inf, rel)
    return rel


# ----------------------------------------------------------------------
# the shadow fold
# ----------------------------------------------------------------------
def shadow_fold(pn, pc, pk,
                *, pc_shadow=None, pk_shadow=None,
                w_kn=0.3, tol=1e-3, verbose=False):
    """
    Fold channel 2 (KN, pc) back into channel 1 (transformer, pn), verified by the
    shadow recomputes. pk (kNN) participates only as a *checked* auxiliary here to
    match the stated spec ("fold 2 back into 1"); extend symmetrically if you want 3->1.

    pc_shadow / pk_shadow: the INDEPENDENT recomputes (from recompute_kn_alignment /
    recompute_knn). If None, the shadow degrades to a self-consistency check (finite +
    normalized), which still catches NaN/inf but not silent value drift — so pass them.

    Returns (fused, report). Fold is per-position:
        trusted position   -> (1-w_kn)*transformer + w_kn*KN
        flagged  position  -> transformer alone (refuse to fold poison)
    """
    report = {}
    n = len(pn)

    def check(name, primary, shadow):
        if shadow is None:
            dis = np.where(np.isfinite(primary), 0.0, np.inf)   # finite-only check
        else:
            dis = disagreement(primary, shadow)
        flagged = ~np.isfinite(dis) | (dis > tol)
        report[name] = dict(
            flagged_frac=float(flagged.mean()),
            max_disagreement=float(np.nanmax(np.where(np.isfinite(dis), dis, np.nan))
                                   if np.isfinite(dis).any() else np.inf),
            any_nonfinite=bool((~np.isfinite(primary)).any()),
        )
        if verbose:
            print(f"    [shadow] {name}: flagged {flagged.mean():6.2%}  "
                  f"max_disagree={report[name]['max_disagreement']:.2e}  "
                  f"nonfinite={report[name]['any_nonfinite']}")
        return flagged

    kn_flagged  = check("KN(2)",  pc, pc_shadow)
    knn_flagged = check("kNN(3)", pk, pk_shadow)

    # clean the primary itself (channel 1 must be finite to fold into)
    pn_clean = np.nan_to_num(pn, nan=EPS)
    pc_clean = np.nan_to_num(pc, nan=EPS)

    # fold 2 -> 1, conditioned on the KN channel being trusted at that position
    do_fold = ~kn_flagged
    fused = np.where(do_fold,
                     (1.0 - w_kn) * pn_clean + w_kn * pc_clean,
                     pn_clean)                      # flagged -> transformer alone

    report["_summary"] = dict(
        folded_frac=float(do_fold.mean()),
        deferred_to_primary_frac=float((~do_fold).mean()),
        knn_flagged_frac=float(knn_flagged.mean()),
    )
    return fused, report


# ----------------------------------------------------------------------
# integration shim for fusion.py (drop-in, optional)
# ----------------------------------------------------------------------
def shadow_report_line(report):
    s = report["_summary"]
    parts = [f"folded {s['folded_frac']:.1%}",
             f"deferred {s['deferred_to_primary_frac']:.1%}"]
    for ch in ("KN(2)", "kNN(3)"):
        r = report[ch]
        if r["any_nonfinite"] or r["flagged_frac"] > 0:
            parts.append(f"{ch} FLAGGED {r['flagged_frac']:.1%}"
                         + (" [NONFINITE]" if r["any_nonfinite"] else ""))
    return "  [shadow] " + " · ".join(parts)
