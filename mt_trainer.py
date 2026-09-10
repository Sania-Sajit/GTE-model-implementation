"""
mt_trainer.py
─────────────
Fine-tunes GTE-Mini on cross-lingual parallel pairs (English ↔ German or English ↔ Hindi)
using GTE's 4-direction Improved Contrastive Loss (ICL).

Default save outputs:
  gte_mt_model/     (for en-de)
  gte_mt_hi_model/  (for en-hi)
"""

import os
import sys
import time
import json
import argparse

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from gte_model import GTEEncoder, improved_contrastive_loss
from mt_dataset_loader import build_mt_dataloader


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_mt(
    base_model_path: str = "gte_mini_model",
    steps: int = 200,
    batch_size: int = 16,
    lr: float = 2e-5,
    max_samples: int = 10000,
    lang_pair: str = "en-de",
    save_dir: str = None,
    save_steps: int = 20,
    log_steps: int = 20,
):
    if save_dir is None:
        save_dir = "gte_mt_hi_model" if lang_pair == "en-hi" else "gte_mt_model"

    ckpt_base = f"checkpoints/mt_{lang_pair.replace('-', '_')}"
    os.makedirs(ckpt_base, exist_ok=True)

    device = get_device()
    print("=" * 65)
    print(f"GTE Phase 4: Machine Translation Fine-Tuning ({lang_pair.upper()})")
    print("=" * 65)
    print(f"  Device           : {device}")
    print(f"  Base Model       : {base_model_path}")
    print(f"  Language Pair    : {lang_pair}")
    print(f"  Steps            : {steps}")
    print(f"  Batch Size       : {batch_size}")
    print(f"  Learning Rate    : {lr}")
    print(f"  Max Samples      : {max_samples}")
    print(f"  Save Steps       : {save_steps}")
    print(f"  Save Directory   : {save_dir}")
    print("=" * 65 + "\n")

    # Load Base Model & Tokenizer
    weights_path = os.path.join(base_model_path, "pytorch_model.bin")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = GTEEncoder("bert-base-uncased").to(device)

    if os.path.exists(weights_path):
        print(f"Loading pretrained GTE weights from '{weights_path}' …")
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))

    model.train()

    # DataLoader
    dataloader = build_mt_dataloader(
        tokenizer=tokenizer,
        max_samples=max_samples,
        batch_size=batch_size,
        lang_pair=lang_pair,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    step = 0
    t0 = time.time()
    data_iter = iter(dataloader)
    loss_history = []
    best_loss = float("inf")

    print(f"\n--- Starting Cross-Lingual Contrastive Fine-Tuning ({steps} steps) ---")

    while step < steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        optimizer.zero_grad()

        q_ids = batch["query_input_ids"].to(device)
        q_mask = batch["query_attention_mask"].to(device)
        d_ids = batch["doc_input_ids"].to(device)
        d_mask = batch["doc_attention_mask"].to(device)

        q_emb = model(q_ids, q_mask)
        d_emb = model(d_ids, d_mask)

        loss = improved_contrastive_loss(q_emb, d_emb, tau=0.01)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        step += 1
        loss_val = float(loss.item())
        loss_history.append(loss_val)

        if device.type == "mps" and step % 10 == 0:
            torch.mps.empty_cache()

        state_data = {
            "step": step,
            "current_loss": loss_val,
            "best_loss": min(best_loss, loss_val),
            "loss_history": loss_history,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Check for new lowest loss on EVERY step (independent of save_steps interval)
        if loss_val < best_loss:
            best_loss = loss_val
            state_data["best_loss"] = best_loss
            best_ckpt = os.path.join(ckpt_base, "best_checkpoint")
            os.makedirs(best_ckpt, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(best_ckpt, "pytorch_model.bin"))
            tokenizer.save_pretrained(best_ckpt)
            with open(os.path.join(best_ckpt, "trainer_state.json"), "w") as f:
                json.dump(state_data, f, indent=2)
            print(f"  ★ Step {step:3d}/{steps} | New Best Loss: {best_loss:.4f} → Updated best_checkpoint/", flush=True)

        if step % log_steps == 0 or step == steps:
            dt = time.time() - t0
            eta = (dt / step) * (steps - step) / 60.0
            print(f"  Step {step:3d}/{steps} | Loss: {loss_val:.4f} | Elapsed: {dt:.1f}s | ETA: {eta:.1f}min", flush=True)

        if step % save_steps == 0 or step == steps:
            last_ckpt = os.path.join(ckpt_base, "last_checkpoint")
            os.makedirs(last_ckpt, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(last_ckpt, "pytorch_model.bin"))
            tokenizer.save_pretrained(last_ckpt)
            with open(os.path.join(last_ckpt, "trainer_state.json"), "w") as f:
                json.dump(state_data, f, indent=2)

            print(f"  ✓ Checkpoint saved → {ckpt_base}/last_checkpoint (step={step}, loss={loss_val:.4f})", flush=True)

    # Save best fine-tuned cross-lingual model to save_dir
    os.makedirs(save_dir, exist_ok=True)
    best_ckpt_file = os.path.join(ckpt_base, "best_checkpoint", "pytorch_model.bin")
    if os.path.exists(best_ckpt_file):
        import shutil
        shutil.copy(best_ckpt_file, os.path.join(save_dir, "pytorch_model.bin"))
        print(f"  ✓ Best checkpoint weights copied to '{save_dir}/pytorch_model.bin'")
    else:
        torch.save(model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
    
    tokenizer.save_pretrained(save_dir)

    with open(os.path.join(save_dir, "trainer_state.json"), "w") as f:
        json.dump({"step": steps, "best_loss": best_loss, "loss_history": loss_history, "final_loss": loss_history[-1]}, f, indent=2)

    print(f"\n✓ Best Cross-Lingual GTE Model ({lang_pair.upper()}) saved to '{save_dir}/'", flush=True)


def main():
    parser = argparse.ArgumentParser(description="GTE Phase 4 MT Fine-Tuning")
    parser.add_argument("--base_model_path", type=str, default="gte_mini_model")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_samples", type=int, default=10000)
    parser.add_argument("--save_steps", type=int, default=20)
    parser.add_argument("--log_steps", type=int, default=20)
    parser.add_argument("--lang_pair", type=str, default="en-de", choices=["en-de", "en-hi"])
    parser.add_argument("--save_dir", type=str, default=None)
    args = parser.parse_args()

    train_mt(
        base_model_path=args.base_model_path,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        max_samples=args.max_samples,
        lang_pair=args.lang_pair,
        save_dir=args.save_dir,
        save_steps=args.save_steps,
        log_steps=args.log_steps,
    )


if __name__ == "__main__":
    main()

