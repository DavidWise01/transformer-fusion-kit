"""Download WikiText-103, build a word-level vocab, tokenize train/valid/test to int32 ids."""
import os, re, urllib.request, numpy as np, pickle, pyarrow.parquet as pq
from collections import Counter
import config as C

BASE = "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1"
FILES = {
    "train": ["train-00000-of-00002.parquet", "train-00001-of-00002.parquet"],
    "valid": ["validation-00000-of-00001.parquet"],
    "test":  ["test-00000-of-00001.parquet"],
}
tokre = re.compile(r"[a-z]+")   # word-level; swap for a BPE tokenizer to remove UNK entirely

def download():
    for split, files in FILES.items():
        for f in files:
            dst = os.path.join(C.DATA_DIR, f)
            if not os.path.exists(dst):
                print("downloading", f)
                urllib.request.urlretrieve(f"{BASE}/{f}", dst)

def rows(f):
    return pq.read_table(os.path.join(C.DATA_DIR, f), columns=["text"]).column("text").to_pylist()

def tokens(split):
    for f in FILES[split]:
        for s in rows(f):
            if s:
                for w in tokre.findall(s.lower()):
                    yield w

def main():
    download()
    # ---- vocab from train ----
    print("counting vocab...")
    cnt = Counter()
    for w in tokens("train"):
        cnt[w] += 1
    K = C.VOCAB_SIZE if C.VOCAB_SIZE else len(cnt)
    vocab = {w: i for i, (w, _) in enumerate(cnt.most_common(K))}
    UNK = len(vocab)
    pickle.dump({"vocab": vocab, "unk": UNK}, open(os.path.join(C.DATA_DIR, "vocab.pkl"), "wb"))
    total = sum(cnt.values()); covered = sum(cnt[w] for w in vocab)
    print(f"vocab {len(vocab)}  UNK rate {100*(1-covered/total):.1f}%")
    # ---- encode each split (streaming into a preallocated array) ----
    for split in ("train", "valid", "test"):
        ids = np.fromiter((vocab.get(w, UNK) for w in tokens(split)), dtype=np.int32)
        np.save(os.path.join(C.DATA_DIR, f"{split}_ids.npy"), ids)
        print(f"{split}: {len(ids):,} tokens")

if __name__ == "__main__":
    main()
