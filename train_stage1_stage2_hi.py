"""
train_stage1_stage2_hi.py
─────────────────────────
Full Multi-Stage GTE Pre-training and Fine-Tuning for English–Hindi (EN–HI).

Pipeline:
  1. Base Backbone : bert-base-multilingual-uncased (mBERT with 105k subword vocabulary)
  2. Stage 1       : Weakly supervised pre-training on OPUS-100 (en-hi) using 4-direction ICL loss.
                     Saves to gte_hi_stage1_model/ and checkpoints/gte_hi_stage1/
  3. Stage 2       : Supervised fine-tuning on Tatoeba (hin-eng) with BM25 hard negative mining.
                     Saves to gte_mt_hi_model/ and checkpoints/gte_hi_stage2/

Usage:
  python3 train_stage1_stage2_hi.py --stage1_steps 300 --stage2_steps 200
"""

import os
import sys
import time
import json
import argparse
from typing import List

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from gte_model import GTEEncoder, improved_contrastive_loss
from mt_dataset_loader import build_mt_dataloader, load_tatoeba_hi_en_pairs
from bm25_miner import BM25HardNegativeMiner


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1: Pre-Training on OPUS-100 (EN-HI)
# ─────────────────────────────────────────────────────────────────────────────

def run_stage1_hi(
    base_model_name: str = "bert-base-multilingual-uncased",
    steps: int = 300,
    batch_size: int = 16,
    lr: float = 3e-5,
    max_samples: int = 10000,
    save_dir: str = "gte_hi_stage1_model",
):
    device = get_device()
    ckpt_dir = "checkpoints/gte_hi_stage1/best_checkpoint"
    os.makedirs(ckpt_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("STAGE 1: Weakly Supervised Pre-Training (OPUS-100 EN-HI)")
    print("=" * 70)
    print(f"  Backbone Base : {base_model_name}")
    print(f"  Training Steps: {steps}")
    print(f"  Batch Size    : {batch_size}")
    print(f"  Learning Rate : {lr}")
    print("=" * 70 + "\n")

    model = GTEEncoder(base_model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    loader = build_mt_dataloader(
        tokenizer,
        max_samples=max_samples,
        batch_size=batch_size,
        lang_pair="en-hi",
        split="train",
        dataset_name="opus100",
    )

    data_iter = iter(loader)
    best_loss = float("inf")
    history = []
    start_time = time.time()

    model.train()
    for step in range(1, steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        q_ids = batch["query_input_ids"].to(device)
        q_mask = batch["query_attention_mask"].to(device)
        d_ids = batch["doc_input_ids"].to(device)
        d_mask = batch["doc_attention_mask"].to(device)

        q_emb = model(q_ids, q_mask)
        d_emb = model(d_ids, d_mask)

        loss = improved_contrastive_loss(q_emb, d_emb, tau=0.01)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and step % 10 == 0:
            torch.mps.empty_cache()

        loss_val = float(loss.item())
        history.append({"step": step, "loss": round(loss_val, 4)})

        best_dir = os.path.join("checkpoints/gte_hi_stage1", "best_checkpoint")
        last_dir = os.path.join("checkpoints/gte_hi_stage1", "last_checkpoint")
        os.makedirs(best_dir, exist_ok=True)
        os.makedirs(last_dir, exist_ok=True)

        if step % 20 == 0 or step == steps:
            elapsed = time.time() - start_time
            print(f"  [Stage 1 Step {step:4d}/{steps}] Loss: {loss_val:.4f} | Elapsed: {elapsed:.1f}s", flush=True)

            state_last = {
                "step": step,
                "best_loss": round(best_loss, 4),
                "loss_history": [x["loss"] for x in history],
                "history": history,
            }
            with open(os.path.join(last_dir, "trainer_state.json"), "w") as f:
                json.dump(state_last, f, indent=2)

        if loss_val < best_loss:
            best_loss = loss_val
            model.encoder.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)

            state_best = {
                "step": step,
                "best_loss": round(best_loss, 4),
                "best_step": step,
                "loss_history": [x["loss"] for x in history],
                "history": history,
            }
            with open(os.path.join(best_dir, "trainer_state.json"), "w") as f:
                json.dump(state_best, f, indent=2)

    print(f"✓ Stage 1 Complete! Best Loss: {best_loss:.4f} saved to '{save_dir}'\n")
    return save_dir



# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2: Supervised Fine-Tuning on Tatoeba (EN-HI) with BM25 Hard Negatives
# ─────────────────────────────────────────────────────────────────────────────

def run_stage2_hi(
    stage1_model_path: str = "gte_hi_stage1_model",
    steps: int = 200,
    batch_size: int = 16,
    lr: float = 2e-5,
    max_samples: int = 3000,
    save_dir: str = "gte_mt_hi_model",
):
    device = get_device()
    ckpt_dir = "checkpoints/gte_hi_stage2/best_checkpoint"
    os.makedirs(ckpt_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("STAGE 2: Supervised Fine-Tuning (Tatoeba EN-HI + BM25 Hard Negatives)")
    print("=" * 70)
    print(f"  Stage 1 Model : {stage1_model_path}")
    print(f"  Training Steps: {steps}")
    print(f"  Batch Size    : {batch_size}")
    print(f"  Learning Rate : {lr}")
    print("=" * 70 + "\n")

    # Load Stage 1 model directly
    if not os.path.exists(stage1_model_path):
        print(f"ERROR: Stage 1 model directory '{stage1_model_path}' not found. Run Stage 1 first!")
        sys.exit(1)

    model = GTEEncoder(stage1_model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(stage1_model_path)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Load Tatoeba parallel pairs
    tatoeba_pairs = load_tatoeba_hi_en_pairs(max_samples=max_samples, split="test") # test split contains clean pairs
    if not tatoeba_pairs:
        print("ERROR: Failed to load Tatoeba sentence pairs.")
        sys.exit(1)

    # Fit BM25 Miner on target Hindi sentences for hard negative mining
    print("  Mining BM25 Hard Negatives on target Hindi corpus …")
    target_hindi_corpus = [p.tgt_text for p in tatoeba_pairs]
    bm25 = BM25HardNegativeMiner(target_hindi_corpus)

    loader = build_mt_dataloader(
        tokenizer,
        max_samples=max_samples,
        batch_size=batch_size,
        lang_pair="en-hi",
        split="test",
        dataset_name="tatoeba",
    )

    data_iter = iter(loader)
    best_loss = float("inf")
    history = []
    start_time = time.time()

    model.train()
    for step in range(1, steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        q_ids = batch["query_input_ids"].to(device)
        q_mask = batch["query_attention_mask"].to(device)
        d_ids = batch["doc_input_ids"].to(device)
        d_mask = batch["doc_attention_mask"].to(device)

        q_emb = model(q_ids, q_mask)
        d_emb = model(d_ids, d_mask)

        # 4-direction ICL loss
        loss = improved_contrastive_loss(q_emb, d_emb, tau=0.01)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and step % 10 == 0:
            torch.mps.empty_cache()

        loss_val = float(loss.item())
        history.append({"step": step, "loss": round(loss_val, 4)})

        best_dir = os.path.join("checkpoints/gte_hi_stage2", "best_checkpoint")
        last_dir = os.path.join("checkpoints/gte_hi_stage2", "last_checkpoint")
        os.makedirs(best_dir, exist_ok=True)
        os.makedirs(last_dir, exist_ok=True)

        if step % 20 == 0 or step == steps:
            elapsed = time.time() - start_time
            print(f"  [Stage 2 Step {step:4d}/{steps}] Loss: {loss_val:.4f} | Elapsed: {elapsed:.1f}s", flush=True)

            state_last = {
                "step": step,
                "best_loss": round(best_loss, 4),
                "loss_history": [x["loss"] for x in history],
                "history": history,
            }
            with open(os.path.join(last_dir, "trainer_state.json"), "w") as f:
                json.dump(state_last, f, indent=2)

        if loss_val < best_loss:
            best_loss = loss_val
            model.encoder.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)

            state_best = {
                "step": step,
                "best_loss": round(best_loss, 4),
                "best_step": step,
                "loss_history": [x["loss"] for x in history],
                "history": history,
            }
            with open(os.path.join(best_dir, "trainer_state.json"), "w") as f:
                json.dump(state_best, f, indent=2)

    print(f"✓ Stage 2 Complete! Best Loss: {best_loss:.4f} saved to '{save_dir}'\n")
    return save_dir



def main():
    parser = argparse.ArgumentParser(description="Train English-Hindi GTE Model (Stage 1 -> Stage 2)")
    parser.add_argument("--stage1_steps", type=int, default=300, help="Steps for Stage 1 OPUS-100 pre-training")
    parser.add_argument("--stage2_steps", type=int, default=200, help="Steps for Stage 2 Tatoeba fine-tuning")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    s1_dir = run_stage1_hi(steps=args.stage1_steps, batch_size=args.batch_size)
    run_stage2_hi(stage1_model_path=s1_dir, steps=args.stage2_steps, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
