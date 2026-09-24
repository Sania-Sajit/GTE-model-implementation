"""
train_stage1_stage2_de.py
─────────────────────────
Full Multi-Stage GTE Pre-training and Fine-Tuning for English–German (EN–DE).

Pipeline:
  1. Base Backbone : bert-base-multilingual-uncased (mBERT with 105k subword vocabulary)
  2. Stage 1       : Weakly supervised pre-training on OPUS-100 (en-de) using 4-direction ICL loss.
                     Saves to gte_de_stage1_model/ and checkpoints/gte_de_stage1/
  3. Stage 2       : Supervised fine-tuning on Tatoeba (deu-eng) with BM25 hard negative mining.
                     Saves to gte_mt_de_model/ and checkpoints/gte_de_stage2/

Usage:
  python3 train_stage1_stage2_de.py --stage1_steps 1000 --stage2_steps 500 --batch_size 16
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
from mt_dataset_loader import build_mt_dataloader, load_tatoeba_de_en_pairs
from bm25_miner import BM25HardNegativeMiner


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1: Pre-Training on OPUS-100 (EN-DE)
# ─────────────────────────────────────────────────────────────────────────────

def run_stage1_de(
    base_model_name: str = "bert-base-multilingual-uncased",
    steps: int = 1000,
    batch_size: int = 16,
    lr: float = 3e-5,
    max_samples: int = 20000,
    save_dir: str = "gte_de_stage1_model",
):
    device = get_device()
    ckpt_dir = "checkpoints/gte_de_stage1/best_checkpoint"
    os.makedirs(ckpt_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("STAGE 1: Weakly Supervised Pre-Training (OPUS-100 EN-DE)")
    print("=" * 70)
    print(f"  Backbone Base : {base_model_name}")
    print(f"  Training Steps: {steps}")
    print(f"  Batch Size    : {batch_size}")
    print(f"  Learning Rate : {lr}")
    print("=" * 70 + "\n")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = GTEEncoder(base_model_name).to(device)

    dataloader = build_mt_dataloader(
        tokenizer,
        max_samples=max_samples,
        batch_size=batch_size,
        lang_pair="en-de",
        split="train",
        dataset_name="opus100",
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()

    best_loss = float("inf")
    history = []
    step = 0
    start_time = time.time()

    data_iter = iter(dataloader)

    while step < steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        step += 1
        query_ids = batch["query_input_ids"].to(device)
        query_mask = batch["query_attention_mask"].to(device)
        doc_ids = batch["doc_input_ids"].to(device)
        doc_mask = batch["doc_attention_mask"].to(device)

        q_embs = model(input_ids=query_ids, attention_mask=query_mask)
        d_embs = model(input_ids=doc_ids, attention_mask=doc_mask)

        loss = improved_contrastive_loss(q_embs, d_embs, tau=0.01)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and step % 10 == 0:
            torch.mps.empty_cache()

        loss_val = float(loss.item())
        history.append({"step": step, "loss": round(loss_val, 4)})

        best_dir = os.path.join("checkpoints/gte_de_stage1", "best_checkpoint")
        last_dir = os.path.join("checkpoints/gte_de_stage1", "last_checkpoint")
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
# STAGE 2: Supervised Fine-Tuning (Tatoeba EN-DE + BM25 Hard Negatives)
# ─────────────────────────────────────────────────────────────────────────────

def run_stage2_de(
    stage1_model_path: str = "gte_de_stage1_model",
    steps: int = 500,
    batch_size: int = 16,
    lr: float = 2e-5,
    max_samples: int = 10000,
    save_dir: str = "gte_mt_de_model",
):
    device = get_device()
    ckpt_dir = "checkpoints/gte_de_stage2/best_checkpoint"
    os.makedirs(ckpt_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("STAGE 2: Supervised Fine-Tuning (Tatoeba EN-DE + BM25 Hard Negatives)")
    print("=" * 70)
    print(f"  Stage 1 Model : {stage1_model_path}")
    print(f"  Training Steps: {steps}")
    print(f"  Batch Size    : {batch_size}")
    print(f"  Learning Rate : {lr}")
    print("=" * 70 + "\n")

    tokenizer = AutoTokenizer.from_pretrained(stage1_model_path)
    model = GTEEncoder(stage1_model_path).to(device)

    # 1. Load Tatoeba German-English bitext
    tatoeba_pairs = load_tatoeba_de_en_pairs(max_samples=max_samples, split="test")
    if not tatoeba_pairs:
        print("ERROR: Failed to load Tatoeba German sentence pairs.")
        sys.exit(1)

    target_de_corpus = [p.tgt_text for p in tatoeba_pairs]
    bm25 = BM25HardNegativeMiner(target_de_corpus)

    loader = build_mt_dataloader(
        tokenizer,
        max_samples=max_samples,
        batch_size=batch_size,
        lang_pair="en-de",
        split="test",
        dataset_name="tatoeba",
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()

    best_loss = float("inf")
    history = []
    start_time = time.time()

    best_dir = os.path.join("checkpoints/gte_de_stage2", "best_checkpoint")
    last_dir = os.path.join("checkpoints/gte_de_stage2", "last_checkpoint")
    os.makedirs(best_dir, exist_ok=True)
    os.makedirs(last_dir, exist_ok=True)

    data_iter = iter(loader)

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
    parser = argparse.ArgumentParser(description="Train English-German GTE Model (Stage 1 -> Stage 2)")
    parser.add_argument("--stage1_steps", type=int, default=1000, help="Steps for Stage 1 OPUS-100 pre-training")
    parser.add_argument("--stage2_steps", type=int, default=500, help="Steps for Stage 2 Tatoeba fine-tuning")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    s1_dir = run_stage1_de(steps=args.stage1_steps, batch_size=args.batch_size)
    run_stage2_de(stage1_model_path=s1_dir, steps=args.stage2_steps, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
