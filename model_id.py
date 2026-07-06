"""model_id.py — a checkable fingerprint for a checkpoint, so model identity
lives on disk as a fact, not in anyone's memory.

Answers three SEPARATE questions (collapsing them is what makes "same model?"
unanswerable):

  ARCH   — same architecture?      hash of the shape/dtype signature of the weights
                                    + the config that built them (d_model, layers,
                                    heads, ctx, vocab). Re-init with same config -> same ARCH.
  WEIGHTS— same trained weights?    content hash of the actual tensor bytes.
                                    Any further training / different seed -> different WEIGHTS.
  STATE  — same training state?     step, best-val, and whether an optimizer state is present.

Two checkpoints are:
  IDENTICAL        all three match
  SAME MODEL, DIFFERENT TRAINING   ARCH matches, WEIGHTS differ
  DIFFERENT MODEL  ARCH differs

Usage:
  python model_id.py CKPT.pt                         # print + write CKPT.pt.fingerprint.json
  python model_id.py CKPT_A.pt CKPT_B.pt             # diff two checkpoints
  python model_id.py --data-manifest wt103_data/     # optional: hash the data ids too

Reads a raw torch checkpoint of the form saved by train.py:
    {"model": state_dict, "opt": ..., "step": int, "best": float}
Falls back gracefully if it's a bare state_dict or a safetensors file.
"""
import sys, os, json, hashlib, argparse

# torch is only needed to LOAD; hashing is format-agnostic once tensors are bytes.
try:
    import torch
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False


def _h(b):
    return hashlib.sha256(b).hexdigest()


def _short(hexstr, n=12):
    return hexstr[:n]


# ----------------------------------------------------------------------
# load a checkpoint into a plain {name: (shape, dtype, raw_bytes)} dict
# ----------------------------------------------------------------------
def load_state(path):
    """Return (state_dict, meta) where state_dict is {name: tensor} and meta holds
    any non-weight fields (step/best/has_opt). Handles train.py's dict, a bare
    state_dict, or safetensors."""
    meta = {"step": None, "best": None, "has_opt": False, "source": os.path.basename(path)}

    if path.endswith(".safetensors"):
        from safetensors import safe_open
        sd = {}
        with safe_open(path, framework="pt") as f:
            for k in f.keys():
                sd[k] = f.get_tensor(k)
        return sd, meta

    if not HAVE_TORCH:
        raise RuntimeError("need torch (or safetensors) to load this checkpoint")

    ck = torch.load(path, map_location="cpu")
    if isinstance(ck, dict) and "model" in ck and isinstance(ck["model"], dict):
        meta["step"] = ck.get("step")
        meta["best"] = ck.get("best")
        meta["has_opt"] = "opt" in ck
        return ck["model"], meta
    if isinstance(ck, dict):
        # bare state_dict
        return ck, meta
    raise RuntimeError(f"unrecognized checkpoint structure in {path}")


# ----------------------------------------------------------------------
# the three fingerprints
# ----------------------------------------------------------------------
def fingerprint(path, config_snapshot=None, data_manifest=None):
    sd, meta = load_state(path)

    # deterministic order
    names = sorted(sd.keys())

    # ARCH signature: name + shape + dtype for every tensor (NOT values)
    arch_sig = "\n".join(
        f"{n}|{tuple(sd[n].shape)}|{str(sd[n].dtype)}" for n in names
    )
    arch_hash = _h(arch_sig.encode())

    # if a config snapshot is given (d_model, n_layers, ...), fold it in so two
    # different configs that happen to yield the same shapes still read as different
    if config_snapshot:
        cfg_str = json.dumps(config_snapshot, sort_keys=True)
        arch_hash = _h((arch_hash + "|" + cfg_str).encode())

    # WEIGHTS hash: content of every tensor, in name order. Cast to a canonical
    # dtype byte view so float16/float32 differences are real differences.
    wh = hashlib.sha256()
    param_count = 0
    for n in names:
        t = sd[n].detach().cpu().contiguous()
        wh.update(n.encode())
        wh.update(str(tuple(t.shape)).encode())
        wh.update(t.numpy().tobytes())
        param_count += t.numel()
    weights_hash = wh.hexdigest()

    fp = {
        "arch":    _short(arch_hash),
        "weights": _short(weights_hash),
        "arch_full":    arch_hash,
        "weights_full": weights_hash,
        "n_params": int(param_count),
        "n_tensors": len(names),
        "state": {
            "step": meta["step"],
            "best_val_ppl": meta["best"],
            "has_optimizer": meta["has_opt"],
        },
        "source": meta["source"],
    }
    if config_snapshot:
        fp["config"] = config_snapshot
    if data_manifest:
        fp["data"] = data_manifest
    return fp


# ----------------------------------------------------------------------
# optional: hash the data the model was trained on (content, not filename)
# ----------------------------------------------------------------------
def data_manifest(data_dir):
    import glob
    out = {}
    for f in sorted(glob.glob(os.path.join(data_dir, "*_ids.npy"))
                    + glob.glob(os.path.join(data_dir, "vocab.pkl"))):
        with open(f, "rb") as fh:
            # hash in chunks so we don't load 100M-token arrays into memory
            h = hashlib.sha256()
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        out[os.path.basename(f)] = _short(h.hexdigest())
    return out


def config_from_kit():
    """Pull the architecture-defining config, if importable, without needing torch."""
    try:
        import config as C
        return {
            "preset": getattr(C, "PRESET", None),
            "vocab_size": getattr(C, "VOCAB_SIZE", None),
            **{k: v for k, v in getattr(C, "MODEL", {}).items()},
        }
    except Exception:
        return None


# ----------------------------------------------------------------------
# compare
# ----------------------------------------------------------------------
def verdict(a, b):
    if a["arch_full"] != b["arch_full"]:
        return "DIFFERENT MODEL", "architecture differs (shapes/config)"
    if a["weights_full"] != b["weights_full"]:
        sa, sb = a["state"]["step"], b["state"]["step"]
        detail = "same architecture, different weights"
        if sa is not None and sb is not None and sa != sb:
            detail += f" (step {sa} vs {sb} — one is further trained)"
        return "SAME MODEL, DIFFERENT TRAINING", detail
    return "IDENTICAL", "arch and weights match to the byte"


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="one checkpoint (fingerprint) or two (diff)")
    ap.add_argument("--data-manifest", metavar="DIR",
                    help="also hash the *_ids.npy / vocab.pkl in DIR")
    ap.add_argument("--no-config", action="store_true",
                    help="skip importing config.py (avoid torch import chain)")
    args = ap.parse_args()

    cfg = None if args.no_config else config_from_kit()
    dm = data_manifest(args.data_manifest) if args.data_manifest else None

    fps = []
    for p in args.paths:
        fp = fingerprint(p, config_snapshot=cfg, data_manifest=dm)
        fps.append(fp)
        sidecar = p + ".fingerprint.json"
        json.dump(fp, open(sidecar, "w"), indent=2)
        print(f"\n{p}")
        print(f"  ARCH    {fp['arch']}   ({fp['n_tensors']} tensors, {fp['n_params']:,} params)")
        print(f"  WEIGHTS {fp['weights']}")
        st = fp["state"]
        print(f"  STATE   step={st['step']}  best_val_ppl={st['best_val_ppl']}  "
              f"opt={'yes' if st['has_optimizer'] else 'no'}")
        if fp.get("data"):
            print(f"  DATA    {fp['data']}")
        print(f"  -> wrote {sidecar}")

    if len(fps) == 2:
        v, detail = verdict(fps[0], fps[1])
        print("\n" + "=" * 56)
        print(f"  VERDICT: {v}")
        print(f"           {detail}")
        print("=" * 56)


if __name__ == "__main__":
    main()
