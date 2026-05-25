import torch
import torch.nn as nn
import torch.nn.functional as F


class Router(nn.Module):
    def __init__(self, n_embd, n_experts, top_k, aux_loss_coef=0.01):
        super().__init__()
        self.n_experts    = n_experts
        self.top_k        = top_k
        self.aux_loss_coef = aux_loss_coef
        self.gate         = nn.Linear(n_embd, n_experts, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        x_flat  = x.view(B * T, C)

        logits  = self.gate(x_flat)
        probs   = F.softmax(logits, dim=-1)

        top_k_probs, top_k_idx = torch.topk(probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        aux_loss = self._load_balance_loss(probs, top_k_idx)

        return top_k_probs, top_k_idx, aux_loss

    def _load_balance_loss(self, probs, top_k_idx):
        n_tokens  = probs.shape[0]
        n_experts = self.n_experts

        dispatch = torch.zeros(n_tokens, n_experts, device=probs.device)
        dispatch.scatter_(1, top_k_idx, 1.0)

        f = dispatch.mean(dim=0)
        p = probs.mean(dim=0)

        loss = self.aux_loss_coef * n_experts * (f * p).sum()
        return loss

    def expert_utilization(self, x):
        B, T, C  = x.shape
        x_flat   = x.view(B * T, C)
        logits   = self.gate(x_flat)
        probs    = F.softmax(logits, dim=-1)
        _, top_k_idx = torch.topk(probs, self.top_k, dim=-1)

        counts = torch.zeros(self.n_experts, device=x.device)
        for i in range(self.top_k):
            counts += torch.bincount(top_k_idx[:, i], minlength=self.n_experts).float()

        return counts / counts.sum()
