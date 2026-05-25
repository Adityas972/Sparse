# MoE — Mixture of Experts Transformer

Implements a sparse Mixture of Experts transformer from scratch and compares it against a dense baseline at matched active compute. Trained on character-level language modelling (tiny Shakespeare or synthetic text).

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

Quick test (2 min on CPU):
```bash
python main.py --max_steps 500 --n_layers 2 --n_embd 128 --block_size 64
```

Full experiment (15–30 min on CPU, faster on GPU):
```bash
python main.py --max_steps 3000 --n_layers 6 --n_embd 256
```

## What it does

Trains two transformer variants and compares them:

**MoE transformer** — each FFN layer is replaced by a Mixture of Experts layer. A learned router sends each token to the top-k experts. Only those k experts activate, so active FLOPs per token = k × expert_size. Total parameters = n_experts × expert_size.

**Dense transformer** — standard transformer with a single FFN. Sized so active FLOPs per token match the MoE model (d_ff = top_k × d_expert).

The key comparison: MoE has more total parameters but identical active compute. Does the extra capacity help?

## Architecture

```
Input tokens
    ↓
Embedding (vocab_size × n_embd)
    ↓  × n_layers
┌─────────────────────────────────────┐
│  LayerNorm + CausalSelfAttention    │
│  + residual                         │
│                                     │
│  LayerNorm + MoE/Dense FFN          │
│  + residual                         │
└─────────────────────────────────────┘
    ↓
LayerNorm → LM Head (n_embd × vocab_size)
```

## MoE Layer

```
token → Router (Linear + softmax) → top-k expert indices + weights
      → Expert_i(token) for each selected i
      → weighted sum → output
```

**Router**: linear projection to n_experts logits, softmax, top-k selection, re-normalise selected weights.

**Load balancing loss** (Switch Transformer formulation):
```
L_aux = n_experts × Σ_i (f_i × p_i)
```
where f_i = fraction of tokens routed to expert i, p_i = mean routing probability to expert i. Encourages uniform expert utilisation.

## Parameter matching

| Model | Total params | Active params/token |
|---|---|---|
| MoE | n_experts × d_expert × ... | top_k × d_expert × ... |
| Dense | d_ff × ... | d_ff × ... | 

With default settings (n_experts=8, top_k=2, d_expert=512): MoE has 4× more FFN params than Dense but identical active compute.

## File structure

```
moe/
├── data/
│   └── dataset.py       Char-level dataset, downloads Shakespeare or generates synthetic text
├── model/
│   ├── attention.py     Causal multi-head self-attention
│   ├── router.py        Top-k gating router + load balance loss
│   ├── moe_layer.py     MoE layer, individual Expert, Dense FFN baseline
│   └── transformer.py   Full transformer (MoE and Dense variants)
├── train.py             Training loop with cosine LR schedule
├── evaluate.py          Perplexity, FLOPs, expert utilization plots
├── main.py              Experiment runner
└── requirements.txt
```

## Interview talking points

**Why does MoE work?** The hypothesis is that different tokens benefit from different kinds of processing — syntax vs semantics, rare words vs common words. Experts specialise implicitly through the router's training signal. The load balancing loss prevents all tokens collapsing to one expert.

**What is the load balancing loss doing?** Without it, the router collapses — it learns to always route to one or two experts that happened to be slightly better early in training (the rich-get-richer problem). L_aux penalises uneven utilisation by maximising its product of f_i and p_i, which is minimised when f_i = 1/n for all i.

**Why top-k=2 and not top-k=1?** Top-1 (Switch Transformer) is simpler and faster but training is less stable because the loss gradient flows through only one expert per token. Top-2 gives two gradient paths and is more robust in practice.

**What's the tradeoff vs dense?** MoE requires more memory (all expert weights loaded even if not active), more complex routing infrastructure, and communication overhead in distributed settings (tokens may need to be sent to different devices where their experts live). The gain is more capacity at the same inference FLOPs.
