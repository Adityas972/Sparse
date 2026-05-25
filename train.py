import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def get_lr(step, warmup_steps, max_steps, max_lr, min_lr=1e-5):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step > max_steps:
        return min_lr
    decay = (step - warmup_steps) / (max_steps - warmup_steps)
    coef  = 0.5 * (1.0 + math.cos(math.pi * decay))
    return min_lr + coef * (max_lr - min_lr)


@torch.no_grad()
def estimate_loss(model, val_loader, device, n_batches=20):
    model.eval()
    losses, ce_losses = [], []

    for i, (x, y) in enumerate(val_loader):
        if i >= n_batches:
            break
        x, y = x.to(device), y.to(device)
        _, loss, ce_loss = model(x, y)
        losses.append(loss.item())
        ce_losses.append(ce_loss.item())

    model.train()
    return {
        "loss"     : sum(losses)    / len(losses),
        "ce_loss"  : sum(ce_losses) / len(ce_losses),
        "perplexity": math.exp(sum(ce_losses) / len(ce_losses)),
    }


def train(model, train_ds, val_ds, cfg, device,
          max_steps=2000, batch_size=32, lr=3e-4,
          eval_interval=200, warmup_steps=100,
          grad_clip=1.0):

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              drop_last=True, num_workers=0)

    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  betas=(0.9, 0.95), weight_decay=0.1)

    history   = {"step": [], "train_loss": [], "val_ppl": [], "val_ce": [],
                 "aux_loss": [], "lr": []}
    train_iter = iter(train_loader)
    t0 = time.time()

    for step in range(max_steps):
        current_lr = get_lr(step, warmup_steps, max_steps, lr)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        _, loss, ce_loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        aux = (loss - ce_loss).item()

        if step % eval_interval == 0 or step == max_steps - 1:
            val_metrics = estimate_loss(model, val_loader, device)
            elapsed     = time.time() - t0

            history["step"].append(step)
            history["train_loss"].append(ce_loss.item())
            history["val_ppl"].append(val_metrics["perplexity"])
            history["val_ce"].append(val_metrics["ce_loss"])
            history["aux_loss"].append(aux)
            history["lr"].append(current_lr)

            print(f"step {step:>5} | train_ce {ce_loss.item():.3f} | "
                  f"val_ppl {val_metrics['perplexity']:>7.2f} | "
                  f"aux {aux:.4f} | lr {current_lr:.1e} | {elapsed:.0f}s")

    return history
