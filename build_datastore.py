"""Build the kNN datastore: run the trained transformer over train, store (hidden state -> next token).

This is the kNN-LM datastore (Khandelwal et al. 2020). At full WT103 scale it is large;
DATASTORE_MAX in config caps it. FAISS is used for the index if available (recommended),
else the fusion step falls back to brute-force torch search.
"""
import os, numpy as np, pickle, torch
import config as C
from model import build_model

@torch.no_grad()
def main():
    vocab = pickle.load(open(os.path.join(C.DATA_DIR, "vocab.pkl"), "rb"))
    Vsize = vocab["unk"] + 1
    tr = np.load(os.path.join(C.DATA_DIR, "train_ids.npy"))
    ctx = C.MODEL["ctx"]
    model = build_model(Vsize)
    model.load_state_dict(torch.load(C.CKPT, map_location=C.DEVICE)["model"]); model.eval()
    d = C.MODEL["d_model"]

    cap = C.DATASTORE_MAX or (len(tr) - ctx)
    keys = np.empty((cap, d), dtype=np.float16)
    vals = np.empty((cap,),   dtype=np.int32)
    pos = 0
    # non-overlapping windows over the stream; take all positions' hidden states as keys
    stride = ctx
    for start in range(0, len(tr) - ctx - 1, stride * C.MICRO_BATCH):
        batch_idx = [start + k * stride for k in range(C.MICRO_BATCH) if start + k * stride + ctx + 1 <= len(tr)]
        if not batch_idx: break
        x = torch.stack([torch.from_numpy(tr[i:i+ctx].astype(np.int64)) for i in batch_idx]).to(C.DEVICE)
        nxt = np.stack([tr[i+1:i+1+ctx] for i in batch_idx])
        with torch.autocast("cuda", enabled=C.AMP):
            _, hid = model(x, return_hidden=True)
        hid = hid.float().cpu().numpy()                         # [b, ctx, d]
        b, T, _ = hid.shape
        flat = hid.reshape(b * T, d); vflat = nxt.reshape(b * T)
        take = min(len(flat), cap - pos)
        keys[pos:pos+take] = flat[:take].astype(np.float16)
        vals[pos:pos+take] = vflat[:take]
        pos += take
        if pos >= cap: break
        if (pos // (stride * C.MICRO_BATCH)) % 50 == 0:
            print(f"datastore {pos:,}/{cap:,}")
    keys = keys[:pos]; vals = vals[:pos]
    np.save(C.DS_PATH + ".keys.npy", keys)
    np.save(C.DS_PATH + ".vals.npy", vals)
    print(f"saved datastore: {pos:,} keys, dim {d}")

    # optional FAISS index for fast search
    try:
        import faiss
        index = faiss.IndexFlatL2(d)
        if C.DEVICE == "cuda":
            index = faiss.index_cpu_to_all_gpus(index)
        index.add(keys.astype(np.float32))
        faiss.write_index(faiss.index_gpu_to_cpu(index) if C.DEVICE == "cuda" else index, C.DS_PATH + ".faiss")
        print("built FAISS index:", C.DS_PATH + ".faiss")
    except Exception as e:
        print("FAISS not built (fusion will brute-force search):", e)

if __name__ == "__main__":
    main()
