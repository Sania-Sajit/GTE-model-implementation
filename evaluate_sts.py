"""
evaluate_sts.py
───────────────
Evaluates sentence embedding quality on the STS Benchmark and STS12-16.

Metric: Spearman rank correlation (ρ) between:
    - cosine similarity of model embeddings
    - gold human similarity scores (0–5, normalised to 0–1)

Compares:
    1.  GTE-Mini  (your trained model from gte_mini_model/)
    2.  thenlper/gte-large  (the full HuggingFace checkpoint)

Both models use the SAME evaluation code path for fairness.

Datasets (from GTE paper evaluation, also standard MTEB):
    - STS Benchmark (stsb)  via glue/stsb split on HuggingFace
    - STS12, STS13, STS14, STS15, STS16  via mteb datasets

Usage:
    python3 evaluate_sts.py                          # evaluate both models
    python3 evaluate_sts.py --model_path gte_mini_model/   # GTE-Mini only
    python3 evaluate_sts.py --gte_large_only               # baseline only
    python3 evaluate_sts.py --dataset stsb                 # single dataset
"""

import argparse
import os
from typing import List, Tuple

import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from tabulate import tabulate
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Embedding helpers
# ─────────────────────────────────────────────────────────────────────────────

def embed_with_gte_mini(
    sentences: List[str],
    model_path: str,
    batch_size: int = 64,
    max_length: int = 512,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Load GTE-Mini and embed a list of sentences."""
    from transformers import AutoTokenizer
    from gte_model import GTEEncoder

    weights_path = os.path.join(model_path, "pytorch_model.bin")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"GTE-Mini weights not found at {weights_path}. "
            "Run train.py first to produce a trained model."
        )

    print(f"  Loading GTE-Mini from {model_path} …")
    model = GTEEncoder("bert-base-uncased")
    model.load_state_dict(
        torch.load(weights_path, map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    return model.encode(
        sentences,
        max_length=max_length,
        batch_size=batch_size,
        device=device,
        normalize=True,
    )


def embed_with_gte_large(
    sentences: List[str],
    batch_size: int = 64,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Load thenlper/gte-large and embed a list of sentences."""
    from sentence_transformers import SentenceTransformer

    print("  Loading thenlper/gte-large …")
    model = SentenceTransformer("thenlper/gte-large", device=str(device))

    embeddings = model.encode(
        sentences,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_tensor=True,
    )
    return embeddings.cpu()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Dataset loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_stsb() -> Tuple[List[str], List[str], List[float]]:
    """
    STS Benchmark (test split).
    Returns (sentences1, sentences2, scores_0_to_1)
    """
    from datasets import load_dataset

    print("  Loading STS Benchmark (mteb/stsbenchmark-sts) …")
    try:
        ds = load_dataset("mteb/stsbenchmark-sts", split="test")
        s1 = [ex["sentence1"] for ex in ds]
        s2 = [ex["sentence2"] for ex in ds]
        # MTEB scores are 0-5 or 0-1, handle both
        max_score = max(ex["score"] for ex in ds) if ds else 1.0
        scale = 5.0 if max_score > 1.0 else 1.0
        scores = [float(ex["score"]) / scale for ex in ds]
    except Exception as e:
        print(f"  WARNING: STSB unavailable ({e}). Skipping.")
        return [], [], []

    print(f"  STS-B: {len(s1)} pairs")
    return s1, s2, scores


def load_sts_year(year: int) -> Tuple[List[str], List[str], List[float]]:
    """
    STS12 through STS16 from MTEB/HuggingFace.
    Returns (sentences1, sentences2, scores_0_to_1)
    """
    from datasets import load_dataset

    dataset_name = f"mteb/sts{year}-sts"
    print(f"  Loading STS{year} ({dataset_name}) …")
    try:
        ds = load_dataset(dataset_name, split="test")
    except Exception as e:
        print(f"  WARNING: STS{year} unavailable ({e}). Skipping.")
        return [], [], []

    s1     = [ex["sentence1"] for ex in ds]
    s2     = [ex["sentence2"] for ex in ds]
    # MTEB datasets already have scores in 0–1 range
    scores = [float(ex["score"]) for ex in ds]

    print(f"  STS{year}: {len(s1)} pairs")
    return s1, s2, scores


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Core evaluation function
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_sts_dataset(
    name:       str,
    s1:         List[str],
    s2:         List[str],
    gold:       List[float],
    embedder,                  # callable: sentences → Tensor (N, D), normalised
) -> float:
    """
    Compute Spearman ρ between cosine similarity scores and gold labels.

    Args:
        name     : dataset name (for display)
        s1, s2   : parallel sentence lists
        gold     : gold similarity scores (0–1)
        embedder : function that maps List[str] → Tensor (N, D)

    Returns:
        Spearman ρ  (float)
    """
    if not s1:
        return float("nan")

    all_sentences = s1 + s2
    embs = embedder(all_sentences)   # (2N, D)

    embs1 = embs[:len(s1)]          # (N, D)
    embs2 = embs[len(s1):]          # (N, D)

    # Cosine similarity for each pair (already normalised → dot product)
    cos_sims = (embs1 * embs2).sum(dim=1).cpu().numpy()   # (N,)

    rho, _ = spearmanr(cos_sims, gold)
    return float(rho)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Main evaluation runner
# ─────────────────────────────────────────────────────────────────────────────

def run_sts_evaluation(
    model_path:     str  = "gte_mini_model",
    gte_large_only: bool = False,
    dataset_filter: str  = "all",
    batch_size:     int  = 64,
    device:         torch.device = torch.device("cpu"),
) -> dict:
    """
    Run STS evaluation on all datasets.

    Returns:
        results dict  {dataset_name: {"gte_mini": rho, "gte_large": rho}}
    """

    # ── Datasets to evaluate on ───────────────────────────────────────────────
    datasets = {}

    if dataset_filter in ("all", "stsb"):
        s1, s2, scores = load_stsb()
        if s1:
            datasets["STS-B"] = (s1, s2, scores)

    for year in [12, 13, 14, 15, 16]:
        key = f"STS{year}"
        if dataset_filter in ("all", key.lower()):
            s1, s2, scores = load_sts_year(year)
            if s1:
                datasets[key] = (s1, s2, scores)

    if not datasets:
        print("No datasets loaded. Check your dataset_filter argument.")
        return {}

    results = {}

    # ── GTE-Mini evaluation ───────────────────────────────────────────────────
    if not gte_large_only:
        print(f"\n{'─'*50}")
        print("Evaluating GTE-Mini …")
        print(f"{'─'*50}")

        def gte_mini_embedder(sentences):
            return embed_with_gte_mini(
                sentences, model_path, batch_size=batch_size, device=device
            )

        for name, (s1, s2, gold) in datasets.items():
            print(f"\n  [{name}] embedding {len(s1)*2} sentences …")
            rho = evaluate_sts_dataset(name, s1, s2, gold, gte_mini_embedder)
            results.setdefault(name, {})["GTE-Mini"] = rho
            print(f"  [{name}] Spearman ρ = {rho:.4f}")

    # ── GTE-large evaluation ──────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("Evaluating thenlper/gte-large …")
    print(f"{'─'*50}")

    def gte_large_embedder(sentences):
        return embed_with_gte_large(sentences, batch_size=batch_size, device=device)

    for name, (s1, s2, gold) in datasets.items():
        print(f"\n  [{name}] embedding {len(s1)*2} sentences …")
        rho = evaluate_sts_dataset(name, s1, s2, gold, gte_large_embedder)
        results.setdefault(name, {})["GTE-large"] = rho
        print(f"  [{name}] Spearman ρ = {rho:.4f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Pretty-print results
# ─────────────────────────────────────────────────────────────────────────────

def print_sts_results(results: dict) -> None:
    """Print a formatted comparison table."""
    if not results:
        return

    headers = ["Dataset"] + list(next(iter(results.values())).keys())
    rows = []
    for dataset, scores in results.items():
        row = [dataset] + [
            f"{v:.4f}" if not (v != v) else "N/A"   # handle NaN
            for v in scores.values()
        ]
        rows.append(row)

    # Average row
    models = list(next(iter(results.values())).keys())
    avgs = []
    for model in models:
        vals = [v for d in results.values()
                for k, v in d.items()
                if k == model and v == v]  # skip NaN
        avgs.append(f"{sum(vals)/len(vals):.4f}" if vals else "N/A")
    rows.append(["AVERAGE"] + avgs)

    print(f"\n{'='*60}")
    print("STS Evaluation Results  (Spearman ρ — higher is better)")
    print(f"{'='*60}")
    print(tabulate(rows, headers=headers, tablefmt="github"))
    print()

    # Expected reference points for context
    print("Reference points (from GTE paper / MTEB leaderboard):")
    print("  BERT-base unconditioned  ≈  0.52  (STS-B)")
    print("  GTE-base  (full training) ≈  0.84  (STS-B)")
    print("  GTE-large (full training) ≈  0.87  (STS-B)")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="STS Evaluation for GTE-Mini vs GTE-large")
    parser.add_argument("--model_path",     type=str, default="gte_mini_model",
                        help="Path to trained GTE-Mini model directory")
    parser.add_argument("--gte_large_only", action="store_true",
                        help="Only evaluate GTE-large (skip GTE-Mini)")
    parser.add_argument("--dataset",        type=str, default="all",
                        choices=["all", "stsb", "sts12", "sts13", "sts14", "sts15", "sts16"],
                        help="Which STS dataset to evaluate on")
    parser.add_argument("--batch_size",     type=int, default=64)
    parser.add_argument("--output_json",    type=str, default="checkpoints/sts_results.json",
                        help="Path to save results as JSON")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    device = (torch.device("mps")  if torch.backends.mps.is_available() else
              torch.device("cuda") if torch.cuda.is_available() else
              torch.device("cpu"))
    print(f"Device: {device}")

    results = run_sts_evaluation(
        model_path=args.model_path,
        gte_large_only=args.gte_large_only,
        dataset_filter=args.dataset,
        batch_size=args.batch_size,
        device=device,
    )

    print_sts_results(results)

    if args.output_json:
        import json
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output_json}")
