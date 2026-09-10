"""
train.py
────────
Entry point for GTE two-stage training.

Usage:
    # Fresh start
    python3 train.py

    # Resume from last checkpoint (auto-detected)
    python3 train.py --resume

    # Custom scale for quick testing
    python3 train.py --stage1_steps 100 --stage2_steps 50 --batch_size 16

    # Full small-scale run
    python3 train.py --stage1_steps 5000 --stage2_steps 1000 --batch_size 128

Google Colab usage:
    !python3 train.py --resume --stage1_steps 5000 --batch_size 64

Checkpoint structure:
    checkpoints/
      stage1/
        last_checkpoint/    ← auto-saved every --save_steps steps
        best_checkpoint/    ← auto-saved when loss improves
      stage2/
        last_checkpoint/
        best_checkpoint/

To continue from an interrupted run, just re-run with --resume.
The trainer will automatically find the last checkpoint and continue.
"""

import argparse
import os
import torch
from transformers import AutoTokenizer

from gte_model      import GTEEncoder
from dataset_loader import build_stage1_dataloader, build_stage2_dataloader
from trainer        import train_stage1, train_stage2


# ─────────────────────────────────────────────────────────────────────────────
# CLI arguments
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="GTE two-stage contrastive training (small-scale)"
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--model_name", type=str, default="bert-base-uncased",
        help="Pretrained encoder to initialise from (default: bert-base-uncased)"
    )

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    parser.add_argument(
        "--stage1_steps", type=int, default=5_000,
        help="Number of gradient steps for Stage 1 (paper: 50,000)"
    )
    parser.add_argument(
        "--stage1_lr", type=float, default=2e-5,
        help="Learning rate for Stage 1 (paper: 2e-5)"
    )
    parser.add_argument(
        "--stage1_max_len", type=int, default=128,
        help="Max sequence length for Stage 1 (paper: 128)"
    )
    parser.add_argument(
        "--stage1_max_samples", type=int, default=50_000,
        help="Max samples to load per Stage 1 source dataset"
    )

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    parser.add_argument(
        "--stage2_steps", type=int, default=1_000,
        help="Number of gradient steps for Stage 2 (paper: ~1 epoch)"
    )
    parser.add_argument(
        "--stage2_lr", type=float, default=2e-6,
        help="Learning rate for Stage 2 (paper: 10× smaller than Stage 1)"
    )
    parser.add_argument(
        "--stage2_max_len", type=int, default=512,
        help="Max sequence length for Stage 2 (paper: 512)"
    )
    parser.add_argument(
        "--stage2_max_samples", type=int, default=20_000,
        help="Max samples to load per Stage 2 source dataset"
    )

    # ── Shared training ───────────────────────────────────────────────────────
    parser.add_argument(
        "--batch_size", type=int, default=128,
        help="Batch size (paper Stage 1: 16384; paper Stage 2: 128)"
    )
    parser.add_argument(
        "--tau", type=float, default=0.01,
        help="Temperature τ for ICL loss (paper: 0.01)"
    )
    parser.add_argument(
        "--warmup_ratio", type=float, default=0.05,
        help="Fraction of steps used for LR warmup (paper: 5%%)"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5,
        help="Multinomial sampling exponent α (paper: 0.5)"
    )

    # ── Checkpointing ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--save_steps", type=int, default=25,
        help="Save last_checkpoint every N steps (default: 25)"
    )
    parser.add_argument(
        "--log_steps", type=int, default=25,
        help="Print loss every N steps (default: 25)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Auto-resume from last checkpoint if it exists"
    )

    # ── Output ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--output_dir", type=str, default="gte_mini_model",
        help="Directory to save the final trained model"
    )

    # ── Misc ──────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--num_workers", type=int, default=0,
        help="DataLoader worker processes (0 = main process)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for dataset sampling"
    )
    parser.add_argument(
        "--skip_stage1", action="store_true",
        help="Skip Stage 1 and load from checkpoints/stage1/best_checkpoint"
    )
    parser.add_argument(
        "--skip_stage2", action="store_true",
        help="Skip Stage 2 (only run Stage 1)"
    )

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Device ────────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")   # Apple Silicon
    else:
        device = torch.device("cpu")

    print(f"\n{'='*60}")
    print(f"GTE Small-Scale Training")
    print(f"{'='*60}")
    print(f"  Device      : {device}")
    print(f"  Model       : {args.model_name}")
    print(f"  Stage 1     : {args.stage1_steps} steps, lr={args.stage1_lr}, "
          f"max_len={args.stage1_max_len}")
    print(f"  Stage 2     : {args.stage2_steps} steps, lr={args.stage2_lr}, "
          f"max_len={args.stage2_max_len}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  τ (tau)     : {args.tau}")
    print(f"  α (alpha)   : {args.alpha}")
    print(f"  Resume      : {args.resume}")
    print(f"{'='*60}\n")

    # ── Tokeniser ─────────────────────────────────────────────────────────────
    print(f"Loading tokenizer ({args.model_name}) …")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"Initialising GTEEncoder ({args.model_name}) …")
    model = GTEEncoder(args.model_name).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}\n")

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 1
    # ─────────────────────────────────────────────────────────────────────────

    if args.skip_stage1:
        # Load from best Stage 1 checkpoint
        best_s1 = "checkpoints/stage1/best_checkpoint/pytorch_model.bin"
        if os.path.isfile(best_s1):
            print(f"Skipping Stage 1. Loading weights from {best_s1} …")
            model.load_state_dict(
                torch.load(best_s1, map_location=device, weights_only=True)
            )
        else:
            print("WARNING: --skip_stage1 set but no Stage 1 checkpoint found. "
                  "Running Stage 1 anyway.")
            args.skip_stage1 = False

    if not args.skip_stage1:
        print("Building Stage 1 DataLoader …")
        stage1_dl = build_stage1_dataloader(
            tokenizer=tokenizer,
            max_samples_per_source=args.stage1_max_samples,
            batch_size=args.batch_size,
            max_length=args.stage1_max_len,
            num_workers=args.num_workers,
            alpha=args.alpha,
            seed=args.seed,
        )

        model = train_stage1(
            model=model,
            dataloader=stage1_dl,
            device=device,
            total_steps=args.stage1_steps,
            learning_rate=args.stage1_lr,
            warmup_ratio=args.warmup_ratio,
            tau=args.tau,
            save_steps=args.save_steps,
            log_steps=args.log_steps,
            resume=args.resume,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 2
    # ─────────────────────────────────────────────────────────────────────────

    if not args.skip_stage2:
        print("Building Stage 2 DataLoader …")
        stage2_dl = build_stage2_dataloader(
            tokenizer=tokenizer,
            max_samples_per_source=args.stage2_max_samples,
            batch_size=args.batch_size,
            max_length=args.stage2_max_len,
            num_workers=args.num_workers,
            alpha=args.alpha,
            seed=args.seed,
        )

        model = train_stage2(
            model=model,
            dataloader=stage2_dl,
            device=device,
            total_steps=args.stage2_steps,
            learning_rate=args.stage2_lr,
            warmup_ratio=args.warmup_ratio,
            tau=args.tau,
            save_steps=max(args.save_steps // 5, 50),
            log_steps=max(args.log_steps // 5, 10),
            resume=args.resume,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Save final model
    # ─────────────────────────────────────────────────────────────────────────

    os.makedirs(args.output_dir, exist_ok=True)
    final_path = os.path.join(args.output_dir, "pytorch_model.bin")
    torch.save(model.state_dict(), final_path)

    # Also save the tokenizer alongside for convenience
    tokenizer.save_pretrained(args.output_dir)

    print(f"\n{'='*60}")
    print(f"✓ Final model saved to: {args.output_dir}/")
    print(f"{'='*60}")
    print("\nNext step: run evaluation")
    print("  python3 evaluate_sts.py")
    print("  python3 evaluate_nli.py")


if __name__ == "__main__":
    main()
