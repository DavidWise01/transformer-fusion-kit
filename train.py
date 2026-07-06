"""Train the transformer on WikiText-103.  CUDA + mixed precision + checkpointed + resumable."""
import os, time, math, numpy as np, pickle, torch, torch.nn.functional as F
import config as C
from model import build_model

def get_batch(ids, ctx, micro, device):
    ix = torch.randint(0, len(ids) - ctx - 1, (micro,))
    x = torch.stack([torch.from_numpy(ids[i:i+ctx].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(ids[i+1:i+1+ctx].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def evaluate(model, ids, ctx, device, batches=200):
    model.eval(); tot, n = 0.0, 0
    for _ in range(batches):
        x, y = get_batch(ids, ctx, C.MICRO_BATCH, device)
        with torch.autocast("cuda", enabled=C.AMP):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        tot += loss.item(); n += 1
    model.train(); return math.exp(tot / n)

def main():
    vocab = pickle.load(open(os.path.join(C.DATA_DIR, "vocab.pkl"), "rb"))
    Vsize = vocab["unk"] + 1
    tr = np.load(os.path.join(C.DATA_DIR, "train_ids.npy"))
    va = np.load(os.path.join(C.DATA_DIR, "valid_ids.npy"))
    ctx = C.MODEL["ctx"]
    model = build_model(Vsize)
    opt = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY, betas=(0.9, 0.95))
    scaler = torch.cuda.amp.GradScaler(enabled=C.AMP)
    accum = max(1, C.BATCH_TOKENS // (C.MICRO_BATCH * ctx))
    step, best = 0, float("inf")
    if os.path.exists(C.CKPT):
        ck = torch.load(C.CKPT, map_location=C.DEVICE)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); step = ck["step"]; best = ck.get("best", best)
        print(f"resumed at step {step} (best val ppl {best:.1f})")

    def lr_at(s):
        if s < C.WARMUP: return C.LR * s / C.WARMUP
        t = (s - C.WARMUP) / max(1, C.MAX_STEPS - C.WARMUP)
        return C.LR * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    t0 = time.time()
    while step < C.MAX_STEPS:
        for g in opt.param_groups: g["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        for _ in range(accum):
            x, y = get_batch(tr, ctx, C.MICRO_BATCH, C.DEVICE)
            with torch.autocast("cuda", enabled=C.AMP):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1)) / accum
            scaler.scale(loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
        scaler.step(opt); scaler.update(); step += 1

        if step % C.EVAL_EVERY == 0:
            vppl = evaluate(model, va, ctx, C.DEVICE)
            best = min(best, vppl)
            print(f"step {step:6d} | val ppl {vppl:7.2f} | best {best:7.2f} | {(time.time()-t0)/60:.1f} min")
        if step % C.CKPT_EVERY == 0:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "best": best}, C.CKPT)
    print("done. final checkpoint:", C.CKPT)

if __name__ == "__main__":
    main()
