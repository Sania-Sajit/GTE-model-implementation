"""
trainer.py
──────────
Training loops for GTE's two-stage contrastive learning.

Stage 1  — Weakly supervised, huge batches, short sequences (max_len=128)
Stage 2  — Supervised + hard negatives, small batches, long sequences (max_len=512),
           learning rate reduced 10× from Stage 1

Checkpoint strategy (both stages):
    checkpoints/
      stage1/
        last_checkpoint/    ← saved every `save_steps` steps  (resume from here)
        best_checkpoint/    ← saved when validation loss improves
      stage2/
        last_checkpoint/
        best_checkpoint/

Each checkpoint directory contains:
    - pytorch_model.bin   (model weights)
    - trainer_state.json  (step, epoch, best_loss, loss history)

On restart, the trainer auto-detects the last checkpoint and resumes.

Paper hyperparameters (GTE-base, full scale):
    Stage 1: batch=16384, lr=2e-5, steps=50000, warmup=5%, max_len=128
    Stage 2: batch=128,   lr=2e-6, epochs=1,    max_len=512, group_size=16
"""

import json
import os
import math
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from gte_model import GTEEncoder, improved_contrastive_loss


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    model:       GTEEncoder,
    optimizer:   torch.optim.Optimizer,
    scheduler,
    step:        int,
    epoch:       int,
    loss:        float,
    best_loss:   float,
    loss_history: list,
    checkpoint_dir: str,
) -> None:
    """
    Save model weights + optimizer state + trainer metadata to checkpoint_dir.
    Creates the directory if it doesn't exist.
    """
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # ── Model weights ─────────────────────────────────────────────────────────
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "pytorch_model.bin"))

    # ── Optimizer + scheduler state ───────────────────────────────────────────
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
        },
        os.path.join(checkpoint_dir, "optimizer.pt"),
    )

    # ── Trainer state (human-readable) ────────────────────────────────────────
    state = {
        "step":         step,
        "epoch":        epoch,
        "current_loss": round(loss, 6),
        "best_loss":    round(best_loss, 6),
        "loss_history": [round(l, 6) for l in loss_history[-100:]],  # last 100
        "timestamp":    time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(checkpoint_dir, "trainer_state.json"), "w") as f:
        json.dump(state, f, indent=2)

    print(f"  ✓ Checkpoint saved → {checkpoint_dir}  (step={step}, loss={loss:.4f})")


def load_checkpoint(
    model:     GTEEncoder,
    optimizer: torch.optim.Optimizer,
    scheduler,
    checkpoint_dir: str,
    device:    torch.device,
) -> tuple[int, int, float, list]:
    """
    Load weights + optimizer from checkpoint_dir.

    Returns:
        (step, epoch, best_loss, loss_history)
    """
    weights_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    optim_path   = os.path.join(checkpoint_dir, "optimizer.pt")
    state_path   = os.path.join(checkpoint_dir, "trainer_state.json")

    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"No model weights found at {weights_path}")

    # Load model
    model.load_state_dict(
        torch.load(weights_path, map_location=device, weights_only=True)
    )

    # Load optimizer
    if os.path.isfile(optim_path):
        optim_state = torch.load(optim_path, map_location=device, weights_only=False)
        optimizer.load_state_dict(optim_state["optimizer"])
        if scheduler and optim_state.get("scheduler"):
            scheduler.load_state_dict(optim_state["scheduler"])

    # Load trainer state
    step, epoch, best_loss, loss_history = 0, 0, float("inf"), []
    if os.path.isfile(state_path):
        with open(state_path) as f:
            state = json.load(f)
        step         = state.get("step",         0)
        epoch        = state.get("epoch",        0)
        best_loss    = state.get("best_loss",    float("inf"))
        loss_history = state.get("loss_history", [])

    print(f"  ✓ Checkpoint loaded ← {checkpoint_dir}  (step={step}, best_loss={best_loss:.4f})")
    return step, epoch, best_loss, loss_history


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Returns path to last_checkpoint/ inside checkpoint_dir if it exists
    and contains a saved model, otherwise None.
    """
    last = os.path.join(checkpoint_dir, "last_checkpoint")
    if os.path.isfile(os.path.join(last, "pytorch_model.bin")):
        return last
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Training loop (shared by Stage 1 and Stage 2)
# ─────────────────────────────────────────────────────────────────────────────

def train_loop(
    model:           GTEEncoder,
    dataloader:      DataLoader,
    optimizer:       torch.optim.Optimizer,
    scheduler,
    device:          torch.device,
    total_steps:     int,
    tau:             float,
    stage_ckpt_dir:  str,          # e.g. "checkpoints/stage1"
    save_steps:      int   = 25,   # save last_checkpoint every N steps
    log_steps:       int   = 25,   # print loss every N steps
    resume_step:     int   = 0,    # start from this step (0 = fresh)
    best_loss:       float = float("inf"),
    loss_history:    list  = None,
    amp_enabled:     bool  = False,
) -> GTEEncoder:
    """
    Core training loop used for both Stage 1 and Stage 2.

    Args:
        model          : GTEEncoder
        dataloader     : iterable of {query_*, document_*} batches
        optimizer      : AdamW
        scheduler      : linear LR scheduler with warmup
        device         : cpu or cuda
        total_steps    : number of gradient steps to run in total
        tau            : temperature for ICL loss (paper: 0.01)
        stage_ckpt_dir : root dir for this stage's checkpoints
        save_steps     : save last_checkpoint every N steps
        log_steps      : print loss every N steps
        resume_step    : if resuming, skip the first `resume_step` steps
        best_loss      : best loss seen so far (from previous run)
        loss_history   : list of past loss values
        amp_enabled    : whether to use FP16 AMP (requires CUDA)

    Returns:
        The trained model.
    """
    if loss_history is None:
        loss_history = []

    last_ckpt_dir = os.path.join(stage_ckpt_dir, "last_checkpoint")
    best_ckpt_dir = os.path.join(stage_ckpt_dir, "best_checkpoint")

    model.train()
    model.to(device)

    # AMP scaler (only for CUDA) — torch.amp API (PyTorch 2.x+, replaces deprecated torch.cuda.amp)
    scaler = torch.amp.GradScaler("cuda") if (amp_enabled and device.type == "cuda") else None

    data_iter  = iter(dataloader)
    step       = resume_step
    epoch      = 0
    total_loss = 0.0
    loss_val   = 0.0
    t0         = time.time()

    if resume_step >= total_steps:
        print(f"\n{'─'*60}")
        print(f"Stage already completed ({resume_step}/{total_steps} steps). Skipping to next stage.")
        print(f"{'─'*60}\n")
        return model

    print(f"\n{'─'*60}")
    print(f"Training: {total_steps - resume_step} steps to go  "
          f"(resuming from step {resume_step})")
    print(f"{'─'*60}\n")

    while step < total_steps:


        # ── Get next batch (cycle the dataloader) ─────────────────────────────
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch    += 1
            data_iter = iter(dataloader)
            batch     = next(data_iter)

        # ── Move to device ────────────────────────────────────────────────────
        q_ids  = batch["query_input_ids"].to(device)
        q_mask = batch["query_attention_mask"].to(device)
        d_ids  = batch["document_input_ids"].to(device)
        d_mask = batch["document_attention_mask"].to(device)

        optimizer.zero_grad()

        # ── Forward pass ──────────────────────────────────────────────────────
        if scaler is not None:
            with torch.amp.autocast("cuda"):   # replaces deprecated torch.cuda.amp.autocast
                q_embs = model(q_ids, q_mask)   # (B, D)
                d_embs = model(d_ids, d_mask)   # (B, D)
                loss   = improved_contrastive_loss(q_embs, d_embs, tau=tau)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            q_embs = model(q_ids, q_mask)
            d_embs = model(d_ids, d_mask)
            loss   = improved_contrastive_loss(q_embs, d_embs, tau=tau)
            loss.backward()
            # Gradient clipping (good practice; paper uses AdamW which handles this)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if scheduler:
            scheduler.step()

        # Periodically flush MPS memory cache on Apple Silicon GPU to prevent OOM
        if device.type == "mps" and step % 10 == 0:
            torch.mps.empty_cache()

        step       += 1

        loss_val    = loss.item()
        total_loss += loss_val
        loss_history.append(loss_val)

        # ── Logging ───────────────────────────────────────────────────────────
        if step % log_steps == 0:
            avg_loss = total_loss / log_steps
            lr       = optimizer.param_groups[0]["lr"]
            elapsed  = time.time() - t0
            steps_left = total_steps - step
            eta      = (elapsed / step) * steps_left if step > 0 else 0
            print(
                f"  Step {step:>6d}/{total_steps} | "
                f"loss={avg_loss:.4f} | "
                f"lr={lr:.2e} | "
                f"epoch={epoch} | "
                f"ETA={eta/60:.1f}min",
                flush=True
            )
            total_loss = 0.0


        # ── Save last_checkpoint every save_steps ────────────────────────────
        if step % save_steps == 0:
            save_checkpoint(
                model, optimizer, scheduler,
                step=step, epoch=epoch,
                loss=loss_val, best_loss=best_loss,
                loss_history=loss_history,
                checkpoint_dir=last_ckpt_dir,
            )

        # ── Save best_checkpoint ──────────────────────────────────────────────
        if loss_val < best_loss:
            best_loss = loss_val
            save_checkpoint(
                model, optimizer, scheduler,
                step=step, epoch=epoch,
                loss=loss_val, best_loss=best_loss,
                loss_history=loss_history,
                checkpoint_dir=best_ckpt_dir,
            )

    # ── Final save ────────────────────────────────────────────────────────────
    save_checkpoint(
        model, optimizer, scheduler,
        step=step, epoch=epoch,
        loss=loss_val, best_loss=best_loss,
        loss_history=loss_history,
        checkpoint_dir=last_ckpt_dir,
    )

    print(f"\n{'─'*60}")
    print(f"Training complete. Best loss: {best_loss:.4f}")
    print(f"{'─'*60}\n")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Stage 1 trainer
# ─────────────────────────────────────────────────────────────────────────────

def train_stage1(
    model:          GTEEncoder,
    dataloader:     DataLoader,
    device:         torch.device,
    total_steps:    int   = 5_000,    # paper: 50,000 (scaled down here)
    learning_rate:  float = 2e-5,     # paper: 2e-5 for GTE-base
    warmup_ratio:   float = 0.05,     # paper: 5% of total steps
    tau:            float = 0.01,
    save_steps:     int   = 500,
    log_steps:      int   = 50,
    amp_enabled:    bool  = False,
    resume:         bool  = True,     # auto-resume if checkpoint exists
) -> GTEEncoder:
    """
    Stage 1: large-scale weakly supervised contrastive pre-training.

    GTE paper (full scale, GTE-base):
        - ~800M pairs
        - batch_size = 16,384
        - max_length = 128
        - steps = 50,000  (~1 epoch)
        - lr = 2e-5, linear decay, 5% warmup
        - AdamW
        - FP16 AMP + DeepSpeed ZeRO Stage 1

    Our small-scale version uses the same hyperparameters but fewer steps
    and smaller batches. All other settings are faithful to the paper.
    """
    print("\n" + "═"*60)
    print("STAGE 1  —  Weakly supervised contrastive pre-training")
    print("═"*60)

    ckpt_dir  = "checkpoints/stage1"
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    warmup_steps = int(total_steps * warmup_ratio)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ── Auto-resume ───────────────────────────────────────────────────────────
    resume_step = 0
    best_loss   = float("inf")
    loss_history = []

    if resume:
        last_ckpt = find_latest_checkpoint(ckpt_dir)
        if last_ckpt:
            print(f"\nFound existing checkpoint: {last_ckpt}")
            print("Resuming from checkpoint …")
            resume_step, _, best_loss, loss_history = load_checkpoint(
                model, optimizer, scheduler, last_ckpt, device
            )
            # Fast-forward scheduler
            for _ in range(resume_step):
                scheduler.step()
        else:
            print("\nNo existing checkpoint found. Starting fresh.")

    model = train_loop(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        total_steps=total_steps,
        tau=tau,
        stage_ckpt_dir=ckpt_dir,
        save_steps=save_steps,
        log_steps=log_steps,
        resume_step=resume_step,
        best_loss=best_loss,
        loss_history=loss_history,
        amp_enabled=amp_enabled,
    )

    return model


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Stage 2 trainer
# ─────────────────────────────────────────────────────────────────────────────

def train_stage2(
    model:          GTEEncoder,
    dataloader:     DataLoader,
    device:         torch.device,
    total_steps:    int   = 1_000,    # paper: ~1 epoch of 3M pairs
    learning_rate:  float = 2e-6,     # paper: LR reduced 10× from Stage 1
    warmup_ratio:   float = 0.05,
    tau:            float = 0.01,
    save_steps:     int   = 200,
    log_steps:      int   = 20,
    amp_enabled:    bool  = False,
    resume:         bool  = True,
) -> GTEEncoder:
    """
    Stage 2: supervised fine-tuning with hard negatives.

    GTE paper (full scale):
        - ~3M supervised pairs
        - batch_size = 128
        - max_length = 512
        - train_group_size = 16  (1 positive + 15 hard negatives per query)
        - lr = 2e-6  (10× smaller than Stage 1)
        - 1 epoch
    """
    print("\n" + "═"*60)
    print("STAGE 2  —  Supervised fine-tuning with hard negatives")
    print("═"*60)

    ckpt_dir  = "checkpoints/stage2"
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    warmup_steps = int(total_steps * warmup_ratio)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ── Auto-resume ───────────────────────────────────────────────────────────
    resume_step  = 0
    best_loss    = float("inf")
    loss_history = []

    if resume:
        last_ckpt = find_latest_checkpoint(ckpt_dir)
        if last_ckpt:
            print(f"\nFound existing checkpoint: {last_ckpt}")
            resume_step, _, best_loss, loss_history = load_checkpoint(
                model, optimizer, scheduler, last_ckpt, device
            )
            for _ in range(resume_step):
                scheduler.step()
        else:
            print("\nNo existing Stage 2 checkpoint. Starting Stage 2 fresh.")

    model = train_loop(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        total_steps=total_steps,
        tau=tau,
        stage_ckpt_dir=ckpt_dir,
        save_steps=save_steps,
        log_steps=log_steps,
        resume_step=resume_step,
        best_loss=best_loss,
        loss_history=loss_history,
        amp_enabled=amp_enabled,
    )

    return model
