import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_heads == 0

        self.n_heads  = cfg.n_heads
        self.n_embd   = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_heads
        self.dropout  = cfg.dropout

        self.qkv  = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size))
            .view(1, 1, cfg.block_size, cfg.block_size)
        )

    def forward(self, x):
        B, T, C = x.shape

        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        att   = (q @ k.transpose(-2, -1)) * scale
        att   = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att   = F.softmax(att, dim=-1)
        att   = self.attn_drop(att)

        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(out))
