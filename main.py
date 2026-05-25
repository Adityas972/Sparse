import torch
import argparse
from data.dataset import make_splits
from model.transformer import Transformer, TransformerConfig
from train import train
from evaluate import print_model_summary, plot_results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--max_steps",      type=int,   default=2000)
    p.add_argument("--batch_size",     type=int,   default=32)
    p.add_argument("--lr",             type=float, default=3e-4)
    p.add_argument("--eval_interval",  type=int,   default=200)
    p.add_argument("--n_embd",         type=int,   default=256)
    p.add_argument("--n_heads",        type=int,   default=8)
    p.add_argument("--n_layers",       type=int,   default=4)
    p.add_argument("--n_experts",      type=int,   default=8)
    p.add_argument("--top_k",          type=int,   default=2)
    p.add_argument("--d_expert",       type=int,   default=512)
    p.add_argument("--block_size",     type=int,   default=128)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds, val_ds = make_splits(block_size=args.block_size)
    vocab_size = train_ds.vocab_size

    d_ff_dense = args.top_k * args.d_expert

    moe_cfg = TransformerConfig(
        vocab_size   = vocab_size,
        block_size   = args.block_size,
        n_embd       = args.n_embd,
        n_heads      = args.n_heads,
        n_layers     = args.n_layers,
        use_moe      = True,
        n_experts    = args.n_experts,
        top_k        = args.top_k,
        d_expert     = args.d_expert,
        d_ff_dense   = d_ff_dense,
    )

    dense_cfg = TransformerConfig(
        vocab_size   = vocab_size,
        block_size   = args.block_size,
        n_embd       = args.n_embd,
        n_heads      = args.n_heads,
        n_layers     = args.n_layers,
        use_moe      = False,
        d_ff_dense   = d_ff_dense,
    )

    moe_model   = Transformer(moe_cfg)
    dense_model = Transformer(dense_cfg)

    print_model_summary(moe_model,   moe_cfg,   "MoE Transformer")
    print_model_summary(dense_model, dense_cfg, "Dense Transformer")

    print("\n── Training MoE ────────────────────────────────────────")
    moe_history = train(
        moe_model, train_ds, val_ds, moe_cfg, device,
        max_steps=args.max_steps, batch_size=args.batch_size,
        lr=args.lr, eval_interval=args.eval_interval,
    )

    print("\n── Training Dense ──────────────────────────────────────")
    dense_history = train(
        dense_model, train_ds, val_ds, dense_cfg, device,
        max_steps=args.max_steps, batch_size=args.batch_size,
        lr=args.lr, eval_interval=args.eval_interval,
    )

    print("\n── Plotting results ─────────────────────────────────────")
    plot_results(moe_history, dense_history, moe_model, dense_model,
                 moe_cfg, dense_cfg)


if __name__ == "__main__":
    main()
