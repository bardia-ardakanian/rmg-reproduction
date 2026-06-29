"""
RMG-base backbone: a frame-token Diffusion Transformer.

Each frame is one token: x (B, L, 91) = [trans(3) | 22 quats (4 each)] -> Linear to hidden D.
Self-attention over the L frames (with a key-padding mask for variable length), AdaLN conditioning from
the time embedding FUSED with the Qwen text vector via an MLP (the paper's HumanML3D recipe). CFG via a
learned null-text token + per-sample dropout. Zero-init output head -> clean flow-matching start.

forward: x (B,L,91), t (B,), text (B,1024) or None, mask (B,L) bool(valid), drop_mask (B,) bool
     ->  velocity (B,L,91)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

IN_DIM = 91
TEXT_DIM = 1024


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        half = dim // 2
        self.register_buffer("freq", torch.exp(-math.log(10000.0) * torch.arange(half) / max(half - 1, 1)))

    def forward(self, t):
        e = t[:, None] * self.freq[None]
        return torch.cat([e.sin(), e.cos()], dim=-1)


class AdaLN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm_a = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm_f = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Linear(dim, 6 * dim)
        nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)

    def forward(self, x, c):
        a_s, a_sh, a_g, f_s, f_sh, f_g = self.proj(c).chunk(6, -1)
        nx = self.norm_a(x) * (1 + a_s[:, None]) + a_sh[:, None]
        nf = self.norm_f(x) * (1 + f_s[:, None]) + f_sh[:, None]
        return nx, nf, a_g[:, None], f_g[:, None]


class SwiGLU(nn.Module):
    def __init__(self, dim, ff):
        super().__init__()
        self.pin = nn.Linear(dim, 2 * ff, bias=False)
        self.pout = nn.Linear(ff, dim, bias=False)

    def forward(self, x):
        g, v = self.pin(x).chunk(2, -1)
        return self.pout(F.silu(g) * v)


class Block(nn.Module):
    def __init__(self, dim, heads, ff):
        super().__init__()
        self.ada = AdaLN(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ff = SwiGLU(dim, ff)

    def forward(self, seq, c, key_padding_mask):
        nx, nf, ag, fg = self.ada(seq, c)
        a, _ = self.attn(nx, nx, nx, key_padding_mask=key_padding_mask, need_weights=False)
        a = torch.nan_to_num(a)                          # fully-masked rows -> 0 (safety)
        seq = seq + ag * a
        seq = seq + fg * self.ff(nf)
        return seq


class RMGTransformer(nn.Module):
    def __init__(self, dim=384, num_layers=6, num_heads=8, ff_mult=8, max_T=200, text_dim=TEXT_DIM):
        super().__init__()
        self.cfg = dict(dim=dim, num_layers=num_layers, num_heads=num_heads,
                        ff_mult=ff_mult, max_T=max_T, text_dim=text_dim)
        self.dim = dim
        self.in_proj = nn.Linear(IN_DIM, dim)
        self.frame_pos = nn.Parameter(torch.zeros(1, max_T, dim)); nn.init.normal_(self.frame_pos, std=0.02)
        self.time_emb = SinusoidalEmbedding(dim)
        self.time_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.text_proj = nn.Linear(text_dim, dim)
        self.null_text = nn.Parameter(torch.zeros(text_dim))
        self.fuse = nn.Sequential(nn.Linear(2 * dim, dim), nn.SiLU(), nn.Linear(dim, dim))   # fuse time + text
        self.blocks = nn.ModuleList([Block(dim, num_heads, ff_mult * dim) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, IN_DIM); nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, x, t, text=None, mask=None, drop_mask=None):
        B, L, _ = x.shape
        tf = self.time_mlp(self.time_emb(t))                          # (B,D)
        if text is None:
            txt = self.null_text[None].expand(B, -1)
        else:
            txt = text
            if drop_mask is not None:
                txt = torch.where(drop_mask[:, None], self.null_text[None], txt)
        c = self.fuse(torch.cat([tf, self.text_proj(txt)], dim=-1))   # (B,D) fused conditioning

        h = self.in_proj(x) + self.frame_pos[:, :L]
        kpm = (~mask) if mask is not None else None                   # True = ignore
        for blk in self.blocks:
            h = blk(h, c, kpm)
        return self.head(self.final_norm(h))

    def get_config(self):
        return dict(self.cfg)
