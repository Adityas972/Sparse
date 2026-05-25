import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from pathlib import Path


def count_flops_per_token(cfg):
    T    = cfg.block_size
    C    = cfg.n_embd
    H    = cfg.n_heads
    L    = cfg.n_layers

    attn_flops = L * (4 * C * C + 2 * T * C)

    if cfg.use_moe:
        ffn_flops = L * cfg.top_k * 2 * C * cfg.d_expert
    else:
        ffn_flops = L * 2 * C * cfg.d_ff_dense

    total = attn_flops + ffn_flops
    return {"attention": attn_flops, "ffn": ffn_flops, "total": total}


def print_model_summary(model, cfg, name):
    params  = model.param_count()
    flops   = count_flops_per_token(cfg)
    print(f"\n{name}")
    print(f"  Total params      : {params:>12,}")
    print(f"  Active FLOPs/token: {flops['total']:>12,}")
    if cfg.use_moe:
        print(f"  Experts           : {cfg.n_experts}  top_k={cfg.top_k}")
        total_expert_params = cfg.n_experts * 2 * cfg.n_embd * cfg.d_expert * cfg.n_layers
        print(f"  Expert params     : {total_expert_params:>12,}  ({total_expert_params/params*100:.1f}% of total)")


def plot_results(moe_history, dense_history, moe_model, dense_model,
                 moe_cfg, dense_cfg, save_dir="results"):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    fig = plt.figure(figsize=(20, 16))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

    moe_color   = "#4C72B0"
    dense_color = "#E07B54"

    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(moe_history["step"],   moe_history["val_ppl"],   lw=2, color=moe_color,   label="MoE")
    ax1.plot(dense_history["step"], dense_history["val_ppl"], lw=2, color=dense_color, label="Dense", linestyle="--")
    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Validation perplexity (lower is better)")
    ax1.set_title("Perplexity Curves: MoE vs Dense")
    ax1.legend()

    ax2 = fig.add_subplot(gs[0, 2])
    moe_flops   = count_flops_per_token(moe_cfg)
    dense_flops = count_flops_per_token(dense_cfg)
    moe_ppl     = moe_history["val_ppl"][-1]
    dense_ppl   = dense_history["val_ppl"][-1]

    ax2.scatter(moe_flops["total"],   moe_ppl,   s=200, color=moe_color,
                marker="o", zorder=5, label=f"MoE  (ppl={moe_ppl:.1f})")
    ax2.scatter(dense_flops["total"], dense_ppl, s=200, color=dense_color,
                marker="s", zorder=5, label=f"Dense (ppl={dense_ppl:.1f})")
    ax2.set_xlabel("Active FLOPs per token")
    ax2.set_ylabel("Final validation perplexity")
    ax2.set_title("Compute efficiency")
    ax2.legend()

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(moe_history["step"], moe_history["train_loss"],
             lw=2, color=moe_color, label="MoE")
    ax3.plot(dense_history["step"], dense_history["train_loss"],
             lw=2, color=dense_color, label="Dense", linestyle="--")
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Train cross-entropy loss")
    ax3.set_title("Training Loss")
    ax3.legend()

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(moe_history["step"], moe_history["aux_loss"],
             lw=1.5, color=moe_color, alpha=0.7)
    ax4.set_xlabel("Step")
    ax4.set_ylabel("Auxiliary (load balance) loss")
    ax4.set_title("MoE Load Balancing Loss")

    ax5 = fig.add_subplot(gs[1, 2])
    utils = moe_model.expert_utilization()
    if utils:
        avg_util = torch.stack(utils).mean(dim=0).numpy()
        n_exp    = len(avg_util)
        colors   = ["#E07B54" if u < 0.5/n_exp else "#4C72B0" for u in avg_util]
        bars     = ax5.bar(range(n_exp), avg_util * 100, color=colors, alpha=0.85)
        ax5.axhline(100.0 / n_exp, color="red", lw=1.5, linestyle="--",
                    label=f"Ideal: {100/n_exp:.1f}%")
        ax5.set_xlabel("Expert index")
        ax5.set_ylabel("% of tokens routed (%)")
        ax5.set_title("Expert Utilization (avg across layers)")
        ax5.legend(fontsize=8)

    ax6 = fig.add_subplot(gs[2, 0])
    metrics = ["Total params", "Active FLOPs/token"]
    moe_vals   = [moe_model.param_count(),   moe_flops["total"]]
    dense_vals = [dense_model.param_count(), dense_flops["total"]]
    x = np.arange(len(metrics))
    w = 0.35
    ax6.bar(x - w/2, moe_vals,   w, label="MoE",   color=moe_color,   alpha=0.85)
    ax6.bar(x + w/2, dense_vals, w, label="Dense", color=dense_color, alpha=0.85)
    ax6.set_xticks(x)
    ax6.set_xticklabels(metrics)
    ax6.set_title("Parameters vs FLOPs")
    ax6.legend()
    ax6.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1e6:.1f}M"))

    ax7 = fig.add_subplot(gs[2, 1:])
    summary = {
        "MoE total params"      : f"{moe_model.param_count():,}",
        "Dense total params"    : f"{dense_model.param_count():,}",
        "MoE active FLOPs"      : f"{moe_flops['total']:,}",
        "Dense active FLOPs"    : f"{dense_flops['total']:,}",
        "MoE final val ppl"     : f"{moe_ppl:.2f}",
        "Dense final val ppl"   : f"{dense_ppl:.2f}",
        "Ppl improvement"       : f"{(dense_ppl - moe_ppl)/dense_ppl*100:.1f}%",
        "MoE param / FLOP ratio": f"{moe_model.param_count()/moe_flops['total']:.2f}x",
    }
    ax7.axis("off")
    rows = list(summary.items())
    ax7.table(
        cellText=[[k, v] for k, v in rows],
        colLabels=["Metric", "Value"],
        cellLoc="left", loc="center",
        colWidths=[0.6, 0.4],
    )
    ax7.set_title("Summary")

    plt.suptitle("MoE vs Dense Transformer — Evaluation Dashboard", fontsize=14, fontweight="bold")
    out = Path(save_dir) / "moe_dashboard.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nDashboard saved to: {out}")
    plt.show()
