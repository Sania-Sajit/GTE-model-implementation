"""
evaluate_nli.py
───────────────
Evaluate sentence embeddings on NLI 3-class classification (SNLI + MultiNLI).

Follows standard Sentence-BERT / GTE evaluation procedure:
  1. Encode premise (u) and hypothesis (v) with frozen GTE encoder
  2. Construct feature vector: [u; v; |u - v|]  (dim: 3 * hidden_size = 2304 for base)
  3. Fit a linear classifier (Logistic Regression) on train split (or small sample)
  4. Evaluate accuracy on test splits (SNLI test, MultiNLI matched/mismatched)

Compares:
  - GTE-Mini (our trained checkpoint)
  - baseline: un-finetuned bert-base-uncased (mean pooled)
  - target: thenlper/gte-large (public HuggingFace GTE checkpoint)

Usage:
  python3 evaluate_nli.py
  python3 evaluate_nli.py --model_path gte_mini_model --batch_size 64
"""

import argparse
import sys
import os
import time
from typing import Tuple, List, Dict

import numpy as np
import torch
import json
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from transformers import AutoModel, AutoTokenizer

from gte_model import GTEEncoder


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Model Wrappers
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_gte_mini(model_dir: str, device: torch.device):
    """Load local GTE-Mini checkpoint."""
    weights_path = os.path.join(model_dir, "pytorch_model.bin")
    if not os.path.exists(weights_path):
        print(f"ERROR: Local model weights '{weights_path}' not found.")
        sys.exit(1)
    print(f"Loading local GTE-Mini from '{model_dir}' …")
    model = GTEEncoder("bert-base-uncased")
    model.load_state_dict(
        torch.load(weights_path, map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer


def load_hf_model(model_name: str, device: torch.device):
    """Load HuggingFace model (e.g. 'thenlper/gte-large' or 'bert-base-uncased')."""
    print(f"Loading HuggingFace model '{model_name}' …")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    backbone = AutoModel.from_pretrained(model_name).to(device)
    backbone.eval()

    class HFWrapper(nn.Module):
        def __init__(self, bb):
            super().__init__()
            self.encoder = bb

        def forward(self, input_ids, attention_mask):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            mask_expanded = attention_mask.unsqueeze(-1).expand(out.last_hidden_state.size()).float()
            sum_embeddings = torch.sum(out.last_hidden_state * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            return torch.nn.functional.normalize(sum_embeddings / sum_mask, p=2, dim=1)

    return HFWrapper(backbone), tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Embedding Generator
# ─────────────────────────────────────────────────────────────────────────────

def encode_sentences(
    sentences: List[str],
    model,
    tokenizer,
    device: torch.device,
    batch_size: int = 64,
    max_length: int = 128,
) -> np.ndarray:
    """Encode a list of sentences into a NumPy array of embeddings."""
    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            batch_texts = sentences[i : i + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

            emb = model(encoded["input_ids"], encoded["attention_mask"])
            all_embeddings.append(emb.cpu().numpy())

    return np.vstack(all_embeddings)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Dataset Loader (SNLI / MultiNLI)
# ─────────────────────────────────────────────────────────────────────────────

LABEL_MAP = {0: 0, 1: 1, 2: 2, "entailment": 0, "neutral": 1, "contradiction": 2}


def load_nli_split(dataset_name: str, config: str, split: str, max_samples: int = 5000) -> Tuple[List[str], List[str], List[int]]:
    """Load premise, hypothesis, label tuples from HF datasets."""
    from datasets import load_dataset

    print(f"  Loading {dataset_name} ({split}) …")
    try:
        if config:
            ds = load_dataset(dataset_name, config, split=split, streaming=True)
        else:
            ds = load_dataset(dataset_name, split=split, streaming=True)
    except Exception as e:
        print(f"  WARNING: Failed to load {dataset_name} ({split}): {e}")
        return [], [], []

    premises, hypotheses, labels = [], [], []
    for ex in ds:
        lbl = ex.get("label", -1)
        if isinstance(lbl, str):
            lbl = LABEL_MAP.get(lbl.lower(), -1)
        if lbl not in (0, 1, 2):
            continue

        p = (ex.get("premise") or ex.get("sentence1") or "").strip()
        h = (ex.get("hypothesis") or ex.get("sentence2") or "").strip()
        if p and h:
            premises.append(p)
            hypotheses.append(h)
            labels.append(lbl)

        if max_samples and len(premises) >= max_samples:
            break

    print(f"  Loaded {len(premises)} pairs")
    return premises, hypotheses, labels


# ─────────────────────────────────────────────────────────────────────────────
# 4.  NLI Evaluation Classifier
# ─────────────────────────────────────────────────────────────────────────────

def train_and_eval_nli(
    model,
    tokenizer,
    device: torch.device,
    train_data: Tuple[List[str], List[str], List[int]],
    test_datasets: Dict[str, Tuple[List[str], List[str], List[int]]],
    batch_size: int = 64,
) -> Dict[str, float]:
    """
    1. Extract u, v embeddings for train premises & hypotheses
    2. Build feature matrix X = [u; v; |u - v|]
    3. Fit Logistic Regression classifier on train split
    4. Predict & report accuracy on test splits
    """
    tr_p, tr_h, tr_y = train_data
    if not tr_p:
        print("  WARNING: Empty training data for NLI evaluation.")
        return {}

    print(f"  Encoding {len(tr_p)} training sentence pairs …")
    u_tr = encode_sentences(tr_p, model, tokenizer, device, batch_size=batch_size)
    v_tr = encode_sentences(tr_h, model, tokenizer, device, batch_size=batch_size)
    X_tr = np.hstack([u_tr, v_tr, np.abs(u_tr - v_tr)])
    y_tr = np.array(tr_y)

    print("  Fitting LogisticRegression classifier …")
    clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs")
    clf.fit(X_tr, y_tr)

    results = {}
    for test_name, (te_p, te_h, te_y) in test_datasets.items():
        if not te_p:
            continue
        print(f"  Encoding {len(te_p)} test pairs for [{test_name}] …")
        u_te = encode_sentences(te_p, model, tokenizer, device, batch_size=batch_size)
        v_te = encode_sentences(te_h, model, tokenizer, device, batch_size=batch_size)
        X_te = np.hstack([u_te, v_te, np.abs(u_te - v_te)])
        preds = clf.predict(X_te)
        acc = float(accuracy_score(te_y, preds)) * 100.0
        results[test_name] = acc
        print(f"  ✓ {test_name} Accuracy: {acc:.2f}%")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Main Driver
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate GTE on NLI benchmarks (SNLI / MultiNLI)")
    parser.add_argument("--model_path", type=str, default="gte_mini_model", help="Path to local GTE-Mini model")
    parser.add_argument("--eval_gte_large", action="store_true", help="Also evaluate public thenlper/gte-large")
    parser.add_argument("--eval_bert_base", action="store_true", help="Also evaluate bert-base-uncased baseline")
    parser.add_argument("--train_samples", type=int, default=5000, help="Number of NLI training pairs for classifier")
    parser.add_argument("--test_samples", type=int, default=1500, help="Number of NLI test pairs per benchmark")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = get_device()
    print("=" * 60)
    print("GTE NLI Evaluation Benchmark (SNLI + MultiNLI)")
    print("=" * 60)
    print(f"  Device        : {device}")
    print(f"  Local Model   : {args.model_path}")
    print(f"  Train Samples : {args.train_samples}")
    print(f"  Test Samples  : {args.test_samples}")
    print("=" * 60 + "\n")

    # Load Datasets
    print("Loading NLI Evaluation Datasets …")
    snli_tr = load_nli_split("stanfordnlp/snli", None, "train", max_samples=args.train_samples)
    snli_te = load_nli_split("stanfordnlp/snli", None, "test", max_samples=args.test_samples)
    mnli_m_te = load_nli_split("nyu-mll/multi_nli", None, "validation_matched", max_samples=args.test_samples)
    mnli_mm_te = load_nli_split("nyu-mll/multi_nli", None, "validation_mismatched", max_samples=args.test_samples)

    test_datasets = {
        "SNLI Test": snli_te,
        "MultiNLI Matched": mnli_m_te,
        "MultiNLI Mismatched": mnli_mm_te,
    }

    all_scores: Dict[str, Dict[str, float]] = {}

    # 1. Local GTE-Mini
    if os.path.exists(args.model_path):
        print(f"\n--- Evaluating local GTE-Mini ({args.model_path}) ---")
        model, tokenizer = load_gte_mini(args.model_path, device)
        all_scores["GTE-Mini (ours)"] = train_and_eval_nli(
            model, tokenizer, device, snli_tr, test_datasets, batch_size=args.batch_size
        )

    # 2. bert-base-uncased baseline
    if args.eval_bert_base:
        print("\n--- Evaluating baseline (bert-base-uncased) ---")
        model_bb, tok_bb = load_hf_model("bert-base-uncased", device)
        all_scores["bert-base-uncased"] = train_and_eval_nli(
            model_bb, tok_bb, device, snli_tr, test_datasets, batch_size=args.batch_size
        )

    # 3. Public gte-large
    if args.eval_gte_large:
        print("\n--- Evaluating target (thenlper/gte-large) ---")
        model_gl, tok_gl = load_hf_model("thenlper/gte-large", device)
        all_scores["gte-large (public)"] = train_and_eval_nli(
            model_gl, tok_gl, device, snli_tr, test_datasets, batch_size=args.batch_size
        )

    # Summary Table
    print("\n" + "=" * 65)
    print("NLI EVALUATION SUMMARY (Accuracy %)")
    print("=" * 65)
    headers = ["Model", "SNLI Test", "MNLI Matched", "MNLI Mismatched"]
    print(f"{headers[0]:<25} | {headers[1]:<10} | {headers[2]:<12} | {headers[3]:<15}")
    print("-" * 65)

    for m_name, score_map in all_scores.items():
        s1 = f"{score_map.get('SNLI Test', 0.0):.2f}%"
        s2 = f"{score_map.get('MultiNLI Matched', 0.0):.2f}%"
        s3 = f"{score_map.get('MultiNLI Mismatched', 0.0):.2f}%"
    print("=" * 65 + "\n")

    # Save to JSON for results_table.py and app.py
    os.makedirs("checkpoints", exist_ok=True)
    out_json = "checkpoints/nli_results.json"
    with open(out_json, "w") as f:
        json.dump(all_scores, f, indent=2)
    print(f"✓ Saved NLI evaluation results to '{out_json}'")


if __name__ == "__main__":
    main()
