"""Plain decoder-only transformer (GPT-style). Shared by train / datastore / fusion.

The hidden state returned alongside logits is the kNN datastore key (the input to the
final output projection), following kNN-LM (Khandelwal et al. 2020).
"""
import math, torch, torch.nn as nn, torch.nn.functional as F
import config as C

class Block(nn.Module):
    def __init__(s, d, h, p=0.1):
        super().__init__()
        s.ln1 = nn.LayerNorm(d); s.ln2 = nn.LayerNorm(d)
        s.att = nn.MultiheadAttention(d, h, batch_first=True, dropout=p)
        s.mlp = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d), nn.Dropout(p))
    def forward(s, x, mask):
        a, _ = s.att(s.ln1(x), s.ln1(x), s.ln1(x), attn_mask=mask, need_weights=False)
        x = x + a
        return x + s.mlp(s.ln2(x))

class GPT(nn.Module):
    def __init__(s, vocab_size, d_model, n_layers, n_heads, ctx):
        super().__init__()
        s.ctx = ctx
        s.tok = nn.Embedding(vocab_size, d_model)
        s.pos = nn.Embedding(ctx, d_model)
        s.drop = nn.Dropout(0.1)
        s.blocks = nn.ModuleList([Block(d_model, n_heads) for _ in range(n_layers)])
        s.lnf = nn.LayerNorm(d_model)
        s.head = nn.Linear(d_model, vocab_size, bias=False)
        s.head.weight = s.tok.weight                       # weight tying
        s.apply(s._init)
    def _init(s, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
    def forward(s, idx, return_hidden=False):
        B, T = idx.shape
        p = torch.arange(T, device=idx.device)
        x = s.drop(s.tok(idx) + s.pos(p)[None])
        mask = torch.triu(torch.full((T, T), float("-inf"), device=idx.device), diagonal=1)
        for b in s.blocks:
            x = b(x, mask)
        hid = s.lnf(x)                                     # <- kNN key
        logits = s.head(hid)
        return (logits, hid) if return_hidden else logits

def build_model(vocab_size):
    m = GPT(vocab_size, **C.MODEL)
    n = sum(p.numel() for p in m.parameters())
    print(f"model: {n/1e6:.1f}M params  ({C.PRESET}, vocab {vocab_size})")
    return m.to(C.DEVICE)
