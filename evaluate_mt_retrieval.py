"""
evaluate_mt_retrieval.py
────────────────────────
Evaluates cross-lingual sentence retrieval performance (Recall@1, Recall@5, Recall@10).
Supports English ↔ German (en-de) and English ↔ Hindi (en-hi).

Usage:
  python3 evaluate_mt_retrieval.py --lang_pair en-de --eval_samples 500
  python3 evaluate_mt_retrieval.py --lang_pair en-hi --eval_samples 500
"""

import os
import sys
import argparse
from typing import List, Tuple, Dict

import numpy as np
import torch
from transformers import AutoTokenizer
import json
from gte_model import GTEEncoder
from mt_dataset_loader import load_tatoeba_pairs


def get_device(device_arg: str = None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_mt_model(model_dir: str, device: torch.device):
    """Load model weights."""
    weights_path = os.path.join(model_dir, "pytorch_model.bin")
    if not os.path.exists(weights_path):
        print(f"ERROR: Model weights not found at '{weights_path}'")
        sys.exit(1)

    model = GTEEncoder("bert-base-uncased")
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    return model



def compute_recall_at_k(similarity_matrix: np.ndarray, ks: List[int] = [1, 5, 10]) -> Dict[int, float]:
    """
    Computes Recall@k given an (N, N) similarity matrix where diagonal elements (i, i)
    are the correct matches.
    """
    N = similarity_matrix.shape[0]
    ranks = np.argsort(-similarity_matrix, axis=1)  # Sort descending

    recalls = {}
    for k in ks:
        top_k = ranks[:, :k]
        targets = np.arange(N)[:, None]
        hits = np.any(top_k == targets, axis=1)
        recalls[k] = float(np.mean(hits)) * 100.0

    return recalls


def evaluate_cross_lingual_retrieval(
    model_dir: str,
    en_sentences: List[str],
    tgt_sentences: List[str],
    device: torch.device,
    batch_size: int = 64,
    lang_tgt: str = "De",
) -> Dict[str, Dict[int, float]]:
    print(f"\n--- Evaluating [{model_dir}] ---")
    model = load_mt_model(model_dir, device)

    print(f"  Encoding {len(en_sentences)} English sentences …")
    en_emb = model.encode(en_sentences, device=device, batch_size=batch_size, normalize=True).cpu().numpy()

    print(f"  Encoding {len(tgt_sentences)} {lang_tgt} sentences …")
    tgt_emb = model.encode(tgt_sentences, device=device, batch_size=batch_size, normalize=True).cpu().numpy()

    # Similarity matrix: (N_en, N_tgt)
    sim_matrix = en_emb @ tgt_emb.T

    # 1. En -> Tgt retrieval
    r_en2tgt = compute_recall_at_k(sim_matrix, ks=[1, 5, 10])

    # 2. Tgt -> En retrieval (transpose similarity matrix)
    r_tgt2en = compute_recall_at_k(sim_matrix.T, ks=[1, 5, 10])

    print(f"  ✓ En → {lang_tgt} Retrieval: R@1={r_en2tgt[1]:.2f}%, R@5={r_en2tgt[5]:.2f}%, R@10={r_en2tgt[10]:.2f}%")
    print(f"  ✓ {lang_tgt} → En Retrieval: R@1={r_tgt2en[1]:.2f}%, R@5={r_tgt2en[5]:.2f}%, R@10={r_tgt2en[10]:.2f}%")

    return {
        f"En->{lang_tgt}": r_en2tgt,
        f"{lang_tgt}->En": r_tgt2en,
    }


def main():
    parser = argparse.ArgumentParser(description="Cross-Lingual MT Retrieval Evaluation")
    parser.add_argument("--base_model", type=str, default="gte_mini_model")
    parser.add_argument("--mt_model", type=str, default=None)
    parser.add_argument("--lang_pair", type=str, default="en-de", choices=["en-de", "en-hi"])
    parser.add_argument("--eval_samples", type=int, default=500)
    parser.add_argument("--device", type=str, default=None, help="Device to run on (cpu, mps, cuda)")
    args = parser.parse_args()

    if args.mt_model is None:
        args.mt_model = "gte_mt_hi_model" if args.lang_pair == "en-hi" else "gte_mt_model"

    lang_tgt = "Hi" if args.lang_pair == "en-hi" else "De"
    device = get_device(args.device)


    print("=" * 65)
    print(f"WMT Cross-Lingual MT Retrieval Evaluation ({args.lang_pair.upper()})")
    print("=" * 65)
    print(f"  Device       : {device}")
    print(f"  Language Pair: {args.lang_pair}")
    print(f"  Eval Samples : {args.eval_samples}")
    print("=" * 65 + "\n")

    # Load Parallel Sentence Pairs (Held-out Test Split)
    pairs = load_tatoeba_pairs(max_samples=args.eval_samples, lang_pair=args.lang_pair, split="test")
    en_texts = [p.src_text for p in pairs]
    tgt_texts = [p.tgt_text for p in pairs]

    all_results = {}

    # 1. Evaluate base GTE-Mini
    if os.path.exists(args.base_model):
        all_results["GTE-Mini (Base)"] = evaluate_cross_lingual_retrieval(
            args.base_model, en_texts, tgt_texts, device, lang_tgt=lang_tgt
        )

    # 2. Evaluate MT Fine-Tuned Model
    if os.path.exists(args.mt_model):
        all_results[f"GTE-MT-{lang_tgt} (Fine-Tuned)"] = evaluate_cross_lingual_retrieval(
            args.mt_model, en_texts, tgt_texts, device, lang_tgt=lang_tgt
        )

    # Print Summary Table
    print("\n" + "=" * 70)
    print(f"CROSS-LINGUAL ({args.lang_pair.upper()}) RETRIEVAL RECALL@K SUMMARY")
    print("=" * 70)
    print(f"{'Model':<25} | {'Direction':<10} | {'R@1':<8} | {'R@5':<8} | {'R@10':<8}")
    print("-" * 70)

    for m_name, direction_map in all_results.items():
        for direction, r_map in direction_map.items():
            r1 = f"{r_map[1]:.2f}%"
            r5 = f"{r_map[5]:.2f}%"
            r10 = f"{r_map[10]:.2f}%"
            print(f"{m_name:<25} | {direction:<10} | {r1:<8} | {r5:<8} | {r10:<8}")

    print("=" * 70 + "\n")

    os.makedirs("checkpoints", exist_ok=True)
    out_json = f"checkpoints/mt_results_{args.lang_pair.replace('-', '_')}.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"✓ Saved MT retrieval results to '{out_json}'")


if __name__ == "__main__":
    main()

