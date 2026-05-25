import torch
import torch.nn as nn
from dataclasses import dataclass
from model.attention import CausalSelfAttention
from model.moe_layer import MoELayer, DenseFFN


@dataclass
class TransformerConfig:
    vocab_size  : int   = 65
    block_size  : int   = 256
    n_embd      : int   = 256
    n_heads     : int   = 8
    n_layers    : int   = 6
    dropout     : float = 0.1
    use_moe     : bool  = True

    n_experts   : int   = 8
    top_k       : int   = 2
    d_expert    : int   = 512
    aux_loss_coef: float= 0.01

    d_ff_dense  : int   = 512

    @property
    def n_params(self):
        n = 0
        n += self.vocab_size * self.n_embd
        n += self.block_size * self.n_embd
        attn_per_layer = 4 * self.n_embd * self.n_embd
        if self.use_moe:
            ffn_per_layer = self.n_experts * 2 * self.n_embd * self.d_expert
        else:
            ffn_per_layer = 2 * self.n_embd * self.d_ff_dense
        n += self.n_layers * (attn_per_layer + ffn_per_layer)
        n += self.n_embd * self.vocab_size
        return n

    @property
    def active_params_per_token(self):
        n = self.vocab_size * self.n_embd + self.block_size * self.n_embd
        attn = 4 * self.n_embd * self.n_embd
        if self.use_moe:
            ffn = self.top_k * 2 * self.n_embd * self.d_expert
        else:
            ffn = 2 * self.n_embd * self.d_ff_dense
        n += self.n_layers * (attn + ffn)
        n += self.n_embd * self.vocab_size
        return n


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1  = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2  = nn.LayerNorm(cfg.n_embd)
        self.ffn  = MoELayer(cfg) if cfg.use_moe else DenseFFN(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        ffn_out, aux_loss = self.ffn(self.ln2(x))
        x = x + ffn_out
        return x, aux_loss


class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg    = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop    = nn.Dropout(cfg.dropout)
        self.blocks  = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f    = nn.LayerNorm(cfg.n_embd)
        self.head    = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, x, targets=None):
        B, T   = x.shape
        device = x.device

        pos    = torch.arange(T, device=device)
        h      = self.drop(self.tok_emb(x) + self.pos_emb(pos))

        aux_loss = torch.tensor(0.0, device=device)
        for block in self.blocks:
            h, block_aux = block(h)
            aux_loss = aux_loss + block_aux

        h    = self.ln_f(h)
        logits = self.head(h)

        loss = None
        if targets is not None:
            import torch.nn.functional as F
            ce_loss  = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss     = ce_loss + aux_loss
            return logits, loss, ce_loss

        return logits, None, None

    @torch.no_grad()
    def generate(self, x, max_new_tokens, temperature=1.0):
        import torch.nn.functional as F
        for _ in range(max_new_tokens):
            x_cond = x[:, -self.cfg.block_size:]
            logits, _, _ = self(x_cond)
            logits = logits[:, -1, :] / temperature
            probs  = F.softmax(logits, dim=-1)
            next_t = torch.multinomial(probs, num_samples=1)
            x      = torch.cat([x, next_t], dim=1)
        return x

    def expert_utilization(self):
        if not self.cfg.use_moe:
            return None
        utils = []
        for block in self.blocks:
            dummy = torch.zeros(1, self.cfg.block_size, self.cfg.n_embd,
                                device=next(self.parameters()).device)
            utils.append(block.ffn.expert_utilization(dummy).cpu())
        return utils

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
