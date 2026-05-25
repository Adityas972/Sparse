import torch
import torch.nn as nn
import torch.nn.functional as F
from model.router import Router


class Expert(nn.Module):
    def __init__(self, n_embd, d_expert, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, d_expert, bias=False),
            nn.GELU(),
            nn.Linear(d_expert, n_embd, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MoELayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_experts = cfg.n_experts
        self.top_k     = cfg.top_k
        self.n_embd    = cfg.n_embd

        self.router  = Router(cfg.n_embd, cfg.n_experts, cfg.top_k, cfg.aux_loss_coef)
        self.experts = nn.ModuleList([
            Expert(cfg.n_embd, cfg.d_expert, cfg.dropout)
            for _ in range(cfg.n_experts)
        ])

    def forward(self, x):
        B, T, C = x.shape
        x_flat  = x.view(B * T, C)

        top_k_probs, top_k_idx, aux_loss = self.router(x)

        out = torch.zeros_like(x_flat)

        for k in range(self.top_k):
            expert_ids   = top_k_idx[:, k]
            expert_probs = top_k_probs[:, k].unsqueeze(1)

            for e in range(self.n_experts):
                mask   = (expert_ids == e)
                if not mask.any():
                    continue
                tokens = x_flat[mask]
                result = self.experts[e](tokens)
                out[mask] += expert_probs[mask] * result

        return out.view(B, T, C), aux_loss

    def expert_utilization(self, x):
        return self.router.expert_utilization(x)


class DenseFFN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, cfg.d_ff_dense, bias=False),
            nn.GELU(),
            nn.Linear(cfg.d_ff_dense, cfg.n_embd, bias=False),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x), torch.tensor(0.0, device=x.device)
