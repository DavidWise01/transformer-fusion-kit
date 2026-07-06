"""uncertainty_gate.py — retrieve only where the transformer is unsure.

The datastore search in fusion.knn_probs is the dominant inference cost: it searches
EVERY position. But kNN-LM only helps where the LM is uncertain; on confident positions
the interpolation barely moves the answer, so that search was paid for nothing.

This gate computes a cheap uncertainty score from the transformer's OWN logits and
searches the datastore only for positions above a threshold. Skipped positions use the
transformer alone. The saving is real because it removes the expensive step, not a
post-hoc mix.

Composes with the shadow channel: the gate makes a SKIP decision, and a wrong skip is a
silent value-fault (a confident position that actually needed retrieval). The shadow
verifies the skip was free by checking the kNN correction on a sample of skipped
positions — if it's large, the threshold is too aggressive.

Nothing here depends on torch; it operates on numpy logits/probs so it can be unit-tested
and calibrated offline, then dropped into fusion.py where logits already exist.
"""
import numpy as np

EPS = 1e-12


# ----------------------------------------------------------------------
# uncertainty signals (all from the LM's own output — no datastore touch)
# ----------------------------------------------------------------------
def uncertainty(probs, kind="entropy"):
    """Per-position uncertainty from the LM distribution `probs` [N, V].
       Higher = less certain = more likely to benefit from retrieval."""
    p = np.clip(probs, EPS, 1.0)
    if kind == "maxprob":
        return 1.0 - p.max(1)                       # cheapest: 1 - top prob
    if kind == "margin":
        top2 = np.partition(p, -2, axis=1)[:, -2:]  # cheap: 1 - (p1 - p2)
        return 1.0 - (top2[:, 1] - top2[:, 0])
    # entropy (default, most principled), normalized to [0,1] by log V
    H = -(p * np.log(p)).sum(1)
    return H / np.log(p.shape[1])


def uncertainty_from_true(pn_true, kind="maxprob_proxy"):
    """Fallback when only the true-token prob is available (fusion.py keeps pn as the
    prob ON the true token, not the full distribution). Uses 1 - p_true as a coarse
    uncertainty proxy: low true-prob positions are where the LM struggled and retrieval
    tends to help. Coarser than full-distribution entropy but free from what fusion has."""
    return 1.0 - np.clip(pn_true, 0, 1)


# ----------------------------------------------------------------------
# the gate: choose which positions search the datastore
# ----------------------------------------------------------------------
def choose(unc, tau=None, budget=None):
    """Return a boolean mask of positions to SEARCH.
       tau:    search where unc > tau.
       budget: alternatively, search the top `budget` fraction (0..1) most-uncertain.
       Exactly one of tau/budget should be set; budget auto-derives the tau."""
    if budget is not None:
        if budget <= 0:   return np.zeros(len(unc), bool)
        if budget >= 1:   return np.ones(len(unc), bool)
        thr = np.quantile(unc, 1.0 - budget)
        return unc > thr
    if tau is None:
        raise ValueError("set tau or budget")
    return unc > tau


# ----------------------------------------------------------------------
# gated fusion: search only masked positions; transformer-alone elsewhere
# ----------------------------------------------------------------------
def gated_fuse(pn, pc, pk_full, mask, w_kn, w_knn):
    """Produce fused probs where `mask` positions get the full triple fusion and
    ~mask positions use transformer(+KN) only — pk is treated as absent (0) there.
    In real use you would NOT compute pk_full off-mask at all (that's the speedup);
    here pk_full is precomputed so we can MEASURE the quality cost of skipping."""
    wn = 1.0 - w_kn - w_knn
    fused = np.where(mask,
                     wn * pn + w_kn * pc + w_knn * pk_full,
                     (1.0 - w_kn) * pn + w_kn * pc)     # off-mask: no kNN term
    return fused


# ----------------------------------------------------------------------
# calibration: the speed/quality Pareto curve (run on a val slice)
# ----------------------------------------------------------------------
def calibrate(pn, pc, pk_full, unc, w_kn, w_knn, ppl_fn, budgets=None):
    """For each search budget, report (skip_frac, ppl). The knee is your operating point.
       Requires pk_full precomputed on the val slice ONCE, to measure what skipping costs."""
    if budgets is None:
        budgets = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
    rows = []
    for b in budgets:
        mask = choose(unc, budget=b)
        fused = gated_fuse(pn, pc, pk_full, mask, w_kn, w_knn)
        rows.append(dict(budget=b, searched_frac=float(mask.mean()),
                         skipped_frac=float((~mask).mean()), ppl=float(ppl_fn(fused))))
    return rows


# ----------------------------------------------------------------------
# shadow check: was the skip actually free? (verify a sample of SKIPPED positions)
# ----------------------------------------------------------------------
def audit_skip(pn, pc, pk_full, mask, w_kn, w_knn, ppl_fn, sample=None, rng=None):
    """The gate's witness. On positions it SKIPPED, compute what the kNN term WOULD have
    changed, and report the PPL cost of skipping them. If large, tau/budget is too tight.
    Sampling keeps this cheap (you don't re-search everything — just an audit fraction)."""
    skipped = np.where(~mask)[0]
    if len(skipped) == 0:
        return dict(audited=0, ppl_with_skip=float("nan"),
                    ppl_if_searched=float("nan"), cost=0.0)
    rng = rng or np.random.default_rng(0)
    if sample and sample < len(skipped):
        skipped = rng.choice(skipped, sample, replace=False)
    wn = 1.0 - w_kn - w_knn
    # what the skipped positions got (transformer+KN only) vs what full fusion gives
    got  = (1.0 - w_kn) * pn[skipped] + w_kn * pc[skipped]
    full = wn * pn[skipped] + w_kn * pc[skipped] + w_knn * pk_full[skipped]
    return dict(
        audited=int(len(skipped)),
        ppl_with_skip=float(ppl_fn(got)),
        ppl_if_searched=float(ppl_fn(full)),
        cost=float(ppl_fn(got) - ppl_fn(full)),   # >0 means skipping HURT: threshold too tight
    )
