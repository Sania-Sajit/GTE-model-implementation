"""
dataset_loader.py
─────────────────
Loads and prepares training data for GTE's two-stage training.

Datasets used are exactly those referenced in the GTE paper:

Stage 1 — Weakly supervised (large scale, short sequences):
  - NLI    : SNLI + MultiNLI  (entailment pairs as positives)
  - Wikipedia : title ↔ first passage
  - MS-MARCO  : query ↔ positive passage
  - Reddit    : post title ↔ body
  - S2ORC     : paper title ↔ abstract

Stage 2 — Supervised with hard negatives (small scale, long sequences):
  - MS-MARCO  + BM25 hard negatives
  - Natural Questions
  - HotpotQA
  - FEVER
  - Quora Question Pairs (QQP)

Sampling strategy (Section 3.4 of paper):
  Multinomial with α = 0.5:
      p_i = n_i^α / Σ n_j^α
  This prevents the largest dataset from dominating.

All batches within a training step come from the SAME source task
(as stated explicitly in the paper, to prevent task-specific shortcuts).
"""

import math
import random
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from transformers import AutoTokenizer


# ─────────────────────────────────────────────────────────────────────────────
# 0.  Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TextPair:
    """A single (query, document) pair used for contrastive training."""
    query:    str
    document: str
    # For Stage 2: optional list of hard-negative document strings
    hard_negatives: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Individual dataset loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_nli_pairs(max_samples: Optional[int] = None) -> List[TextPair]:
    """
    SNLI + MultiNLI  —  entailment pairs only.
    Updated dataset IDs for HuggingFace datasets 5.0:
      snli        → stanfordnlp/snli
      multi_nli   → nyu-mll/multi_nli
    """
    from datasets import load_dataset

    pairs: List[TextPair] = []

    # ── SNLI ──────────────────────────────────────────────────────────────────
    print("  Loading SNLI …")
    snli = load_dataset("stanfordnlp/snli", split="train", streaming=True)
    for ex in snli:
        if ex["label"] == 0:   # 0 = entailment
            pairs.append(TextPair(query=ex["premise"], document=ex["hypothesis"]))
        if max_samples and len(pairs) >= max_samples // 2:
            break

    # ── MultiNLI ──────────────────────────────────────────────────────────────
    print("  Loading MultiNLI …")
    mnli = load_dataset("nyu-mll/multi_nli", split="train", streaming=True)
    for ex in mnli:
        if ex["label"] == 0:   # 0 = entailment
            pairs.append(TextPair(query=ex["premise"], document=ex["hypothesis"]))
        if max_samples and len(pairs) >= max_samples:
            break

    print(f"  NLI pairs loaded: {len(pairs):,}")
    return pairs


def load_wikipedia_pairs(max_samples: Optional[int] = None) -> List[TextPair]:
    """
    Wikipedia title ↔ first passage pairs.
    Updated ID for datasets 5.0: wikimedia/wikipedia
    Uses streaming=True — avoids downloading the full 20GB dump.
    """
    from datasets import load_dataset

    print("  Loading Wikipedia (wikimedia/wikipedia, en, 20231101) …")
    wiki = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=True,
    )

    pairs: List[TextPair] = []
    for ex in wiki:
        title = ex["title"].strip()
        paragraphs = [p.strip() for p in ex["text"].split("\n") if p.strip()]
        if not title or not paragraphs:
            continue
        pairs.append(TextPair(query=title, document=paragraphs[0]))
        if max_samples and len(pairs) >= max_samples:
            break

    print(f"  Wikipedia pairs loaded: {len(pairs):,}")
    return pairs


def load_msmarco_pairs(max_samples: Optional[int] = None) -> List[TextPair]:
    """
    MS-MARCO v2.1  —  query ↔ positive passage.
    Updated ID for datasets 5.0: microsoft/ms_marco
    Uses streaming=True to avoid downloading the full dataset.
    """
    from datasets import load_dataset

    print("  Loading MS-MARCO (microsoft/ms_marco) …")
    ms = load_dataset("microsoft/ms_marco", "v2.1", split="train", streaming=True)

    pairs: List[TextPair] = []
    for ex in ms:
        query     = ex["query"]
        passages  = ex["passages"]["passage_text"]
        is_answer = ex["passages"]["is_selected"]
        for passage, selected in zip(passages, is_answer):
            if selected == 1:
                pairs.append(TextPair(query=query, document=passage))
                break
        if max_samples and len(pairs) >= max_samples:
            break

    print(f"  MS-MARCO pairs loaded: {len(pairs):,}")
    return pairs


def load_reddit_pairs(max_samples: Optional[int] = None) -> List[TextPair]:
    """
    Reddit  —  post title ↔ body.
    Uses streaming=True — avoids downloading multi-GB files for small samples.
    """
    from datasets import load_dataset

    print("  Loading Reddit (sentence-transformers/reddit-title-body) …")
    try:
        reddit = load_dataset(
            "sentence-transformers/reddit-title-body",
            split="train",
            streaming=True,    # CRITICAL: prevents downloading the full dataset
        )
    except Exception as e:
        print(f"  WARNING: Reddit dataset unavailable ({e}). Skipping.")
        return []

    pairs: List[TextPair] = []
    for ex in reddit:
        title = (ex.get("title") or "").strip()
        body  = (ex.get("body")  or "").strip()
        if title and body:
            pairs.append(TextPair(query=title, document=body))
        if max_samples and len(pairs) >= max_samples:
            break

    print(f"  Reddit pairs loaded: {len(pairs):,}")
    return pairs


def load_s2orc_pairs(max_samples: Optional[int] = None) -> List[TextPair]:
    """
    S2ORC  —  paper title ↔ abstract.
    Uses streaming=True to avoid downloading the full corpus.
    """
    from datasets import load_dataset

    print("  Loading S2ORC (sentence-transformers/s2orc) …")
    try:
        s2orc = load_dataset(
            "sentence-transformers/s2orc",
            split="train",
            streaming=True,    # CRITICAL: avoids large download
        )
    except Exception as e:
        print(f"  WARNING: S2ORC dataset unavailable ({e}). Skipping.")
        return []

    pairs: List[TextPair] = []
    for ex in s2orc:
        title    = (ex.get("title")    or "").strip()
        abstract = (ex.get("abstract") or "").strip()
        if title and abstract:
            pairs.append(TextPair(query=title, document=abstract))
        if max_samples and len(pairs) >= max_samples:
            break

    print(f"  S2ORC pairs loaded: {len(pairs):,}")
    return pairs


# ── Stage 2 datasets ──────────────────────────────────────────────────────────

def load_natural_questions_pairs(max_samples: Optional[int] = None) -> List[TextPair]:
    """
    Natural Questions — question ↔ short answer (Stage 2).
    Uses fast streaming sentence-transformers/natural-questions.
    """
    from datasets import load_dataset

    print("  Loading Natural Questions (sentence-transformers/natural-questions) …")
    pairs: List[TextPair] = []

    try:
        nq = load_dataset("sentence-transformers/natural-questions", split="train", streaming=True)
        for ex in nq:
            query = (ex.get("query") or "").strip()
            answer = (ex.get("answer") or "").strip()
            if query and answer:
                pairs.append(TextPair(query=query, document=answer))
            if max_samples and len(pairs) >= max_samples:
                break
    except Exception as e:
        print(f"  WARNING: sentence-transformers NQ unavailable ({e}). Trying fallback.")
        try:
            nq = load_dataset("google-research-datasets/natural_questions", split="train", streaming=True)
            for ex in nq:
                question = ex["question"]["text"].strip()
                short_answers = ex["annotations"]["short_answers"]
                if short_answers and short_answers[0]["text"]:
                    answer = short_answers[0]["text"][0].strip()
                    if question and answer:
                        pairs.append(TextPair(query=question, document=answer))
                if max_samples and len(pairs) >= max_samples:
                    break
        except Exception as e2:
            print(f"  WARNING: NQ fallback failed ({e2}). Skipping.")

    print(f"  Natural Questions pairs loaded: {len(pairs):,}")
    return pairs



def load_hotpotqa_pairs(max_samples: Optional[int] = None) -> List[TextPair]:
    """HotpotQA — multi-hop question ↔ supporting passage (Stage 2)."""
    from datasets import load_dataset

    print("  Loading HotpotQA …")
    hotpot = load_dataset("hotpotqa/hotpot_qa", "fullwiki", split="train",
                          streaming=True)

    pairs: List[TextPair] = []
    for ex in hotpot:
        question   = ex["question"].strip()
        sup_titles = ex["supporting_facts"]["title"]
        context    = {t: s for t, s in zip(ex["context"]["title"],
                                           ex["context"]["sentences"])}
        for title in sup_titles:
            if title in context and context[title]:
                document = " ".join(context[title][:2])
                if question and document:
                    pairs.append(TextPair(query=question, document=document))
                    break
        if max_samples and len(pairs) >= max_samples:
            break

    print(f"  HotpotQA pairs loaded: {len(pairs):,}")
    return pairs


def load_fever_pairs(max_samples: Optional[int] = None) -> List[TextPair]:
    """FEVER — claim ↔ supporting evidence (Stage 2, SUPPORTS only)."""
    from datasets import load_dataset

    print("  Loading FEVER (cais/fever) …")
    try:
        fever = load_dataset("cais/fever", split="train", streaming=True)
    except Exception as e:
        print(f"  WARNING: FEVER unavailable ({e}). Skipping.")
        return []

    pairs: List[TextPair] = []
    for ex in fever:
        label = ex.get("label", "")
        if label == "SUPPORTS":
            claim = ex.get("claim", "").strip()
            evidence = ex.get("evidence_wiki_url", "") or ex.get("evidence", "")
            if isinstance(evidence, str):
                evidence = evidence.strip().replace("_", " ")
            else:
                evidence = ""
            if claim and evidence:
                pairs.append(TextPair(query=claim, document=evidence))
        if max_samples and len(pairs) >= max_samples:
            break

    print(f"  FEVER pairs loaded: {len(pairs):,}")
    return pairs


def load_qqp_pairs(max_samples: Optional[int] = None) -> List[TextPair]:
    """Quora Question Pairs — duplicate question pairs (Stage 2)."""
    from datasets import load_dataset

    print("  Loading Quora Question Pairs (SetFit/qqp) …")
    try:
        qqp = load_dataset("SetFit/qqp", split="train", streaming=True)
    except Exception as e:
        print(f"  WARNING: QQP unavailable ({e}). Skipping.")
        return []

    pairs: List[TextPair] = []
    for ex in qqp:
        is_dup = ex.get("label", 0) == 1 or ex.get("is_duplicate", False)
        if is_dup:
            q1 = (ex.get("text1") or ex.get("question1") or "").strip()
            q2 = (ex.get("text2") or ex.get("question2") or "").strip()
            if q1 and q2:
                pairs.append(TextPair(query=q1, document=q2))
        if max_samples and len(pairs) >= max_samples:
            break

    print(f"  QQP pairs loaded: {len(pairs):,}")
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Multinomial sampler  (GTE paper Section 3.4, α = 0.5)
# ─────────────────────────────────────────────────────────────────────────────

def compute_sampling_probabilities(
    dataset_sizes: List[int],
    alpha: float = 0.5,
) -> List[float]:
    """
    GTE paper Eq. for data sampling:

        p_i = n_i^α / Σ_j n_j^α

    With α = 0.5 this is proportional to √n_i, which reduces the dominance
    of very large datasets while still sampling larger ones more often.

    Args:
        dataset_sizes : list of sizes of each dataset
        alpha         : exponent  (paper uses 0.5)

    Returns:
        list of sampling probabilities (sums to 1.0)
    """
    powered = [n ** alpha for n in dataset_sizes]
    total   = sum(powered)
    return [p / total for p in powered]


# ─────────────────────────────────────────────────────────────────────────────
# 3.  PyTorch Dataset wrappers
# ─────────────────────────────────────────────────────────────────────────────

class PairDataset(Dataset):
    """
    Simple map-style dataset of TextPair objects.
    Tokenises on-the-fly in __getitem__.
    """

    def __init__(
        self,
        pairs:      List[TextPair],
        tokenizer:  AutoTokenizer,
        max_length: int = 128,
    ):
        self.pairs      = pairs
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        pair = self.pairs[idx]
        return {
            "query":    pair.query,
            "document": pair.document,
        }

    def collate_fn(self, batch: list) -> dict:
        """
        Tokenise a batch of (query, document) string pairs.
        Returns a dict with 'query_*' and 'document_*' tensors.
        """
        queries   = [b["query"]    for b in batch]
        documents = [b["document"] for b in batch]

        q_enc = self.tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        d_enc = self.tokenizer(
            documents,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "query_input_ids":          q_enc["input_ids"],
            "query_attention_mask":     q_enc["attention_mask"],
            "document_input_ids":       d_enc["input_ids"],
            "document_attention_mask":  d_enc["attention_mask"],
        }


class MultinomialSampledDataset(IterableDataset):
    """
    Iterates over multiple PairDatasets using GTE's multinomial sampling
    strategy (α = 0.5). Within each step, ALL items come from the SAME
    source dataset (as stated in the paper).

    Yields individual TextPair dicts (collation handled by DataLoader).
    """

    def __init__(
        self,
        datasets:  List[List[TextPair]],   # one list per source
        alpha:     float = 0.5,
        seed:      int   = 42,
    ):
        self.datasets = datasets
        self.probs    = compute_sampling_probabilities(
            [len(d) for d in datasets], alpha=alpha
        )
        self.rng      = random.Random(seed)

    def __iter__(self) -> Iterator[dict]:
        # Build shuffled index lists for each dataset
        indices = [
            list(range(len(d))) for d in self.datasets
        ]
        for idx_list in indices:
            self.rng.shuffle(idx_list)

        pointers = [0] * len(self.datasets)

        while True:
            # Pick a dataset according to multinomial probabilities
            chosen = self.rng.choices(
                range(len(self.datasets)),
                weights=self.probs,
                k=1,
            )[0]

            ptr = pointers[chosen]
            if ptr >= len(indices[chosen]):
                # Reshuffle this dataset when exhausted
                self.rng.shuffle(indices[chosen])
                pointers[chosen] = 0
                ptr = 0

            pair = self.datasets[chosen][indices[chosen][ptr]]
            pointers[chosen] += 1

            yield {
                "query":    pair.query,
                "document": pair.document,
            }


# ─────────────────────────────────────────────────────────────────────────────
# 4.  High-level builders
# ─────────────────────────────────────────────────────────────────────────────

def build_stage1_dataloader(
    tokenizer:   AutoTokenizer,
    max_samples_per_source: int   = 50_000,   # small-scale cap per dataset
    batch_size:  int              = 512,
    max_length:  int              = 128,
    num_workers: int              = 0,
    alpha:       float            = 0.5,
    seed:        int              = 42,
) -> DataLoader:
    """
    Build the Stage 1 DataLoader using GTE paper's Stage 1 sources.

    Paper settings (full scale):
        - ~800M pairs total
        - max_length = 128
        - batch_size > 10,000

    Small-scale settings used here:
        - max_samples_per_source caps each source independently
        - batch_size is a configurable argument
    """
    print("\n[Stage 1] Loading datasets …")
    sources = []

    # Load each source (catches failures gracefully)
    for loader_fn in [
        lambda: load_nli_pairs(max_samples_per_source),
        lambda: load_wikipedia_pairs(max_samples_per_source),
        lambda: load_msmarco_pairs(max_samples_per_source),
        lambda: load_reddit_pairs(max_samples_per_source),
        lambda: load_s2orc_pairs(max_samples_per_source),
    ]:
        try:
            data = loader_fn()
            if data:
                sources.append(data)
        except Exception as e:
            print(f"  WARNING: Source failed to load: {e}")

    print(f"\n[Stage 1] {len(sources)} source(s) loaded.")
    for i, s in enumerate(sources):
        print(f"  Source {i}: {len(s):,} pairs")

    sampled_dataset = MultinomialSampledDataset(
        datasets=sources,
        alpha=alpha,
        seed=seed,
    )

    def collate_fn(batch):
        queries   = [b["query"]    for b in batch]
        documents = [b["document"] for b in batch]
        q_enc = tokenizer(queries,   padding=True, truncation=True,
                          max_length=max_length, return_tensors="pt")
        d_enc = tokenizer(documents, padding=True, truncation=True,
                          max_length=max_length, return_tensors="pt")
        return {
            "query_input_ids":         q_enc["input_ids"],
            "query_attention_mask":    q_enc["attention_mask"],
            "document_input_ids":      d_enc["input_ids"],
            "document_attention_mask": d_enc["attention_mask"],
        }

    return DataLoader(
        sampled_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=False,
    )


def build_stage2_dataloader(
    tokenizer:   AutoTokenizer,
    max_samples_per_source: int   = 20_000,
    batch_size:  int              = 128,
    max_length:  int              = 512,
    num_workers: int              = 0,
    alpha:       float            = 0.5,
    seed:        int              = 42,
) -> DataLoader:
    """
    Build the Stage 2 DataLoader using GTE paper's Stage 2 supervised sources.

    Paper settings (full scale):
        - ~3M supervised pairs
        - max_length = 512
        - batch_size = 128
        - train_group_size = 16 (query + 1 positive + 15 hard negatives)

    Small-scale settings used here:
        - max_samples_per_source caps each source
    """
    print("\n[Stage 2] Loading datasets …")
    sources = []

    for loader_fn in [
        lambda: load_msmarco_pairs(max_samples_per_source),
        lambda: load_natural_questions_pairs(max_samples_per_source),
        lambda: load_hotpotqa_pairs(max_samples_per_source),
        lambda: load_fever_pairs(max_samples_per_source),
        lambda: load_qqp_pairs(max_samples_per_source),
    ]:
        try:
            data = loader_fn()
            if data:
                sources.append(data)
        except Exception as e:
            print(f"  WARNING: Source failed to load: {e}")

    print(f"\n[Stage 2] {len(sources)} source(s) loaded.")
    for i, s in enumerate(sources):
        print(f"  Source {i}: {len(s):,} pairs")

    sampled_dataset = MultinomialSampledDataset(
        datasets=sources,
        alpha=alpha,
        seed=seed,
    )

    def collate_fn(batch):
        queries   = [b["query"]    for b in batch]
        documents = [b["document"] for b in batch]
        q_enc = tokenizer(queries,   padding=True, truncation=True,
                          max_length=max_length, return_tensors="pt")
        d_enc = tokenizer(documents, padding=True, truncation=True,
                          max_length=max_length, return_tensors="pt")
        return {
            "query_input_ids":         q_enc["input_ids"],
            "query_attention_mask":    q_enc["attention_mask"],
            "document_input_ids":      d_enc["input_ids"],
            "document_attention_mask": d_enc["attention_mask"],
        }

    return DataLoader(
        sampled_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Quick sanity check  (run with:  python3 dataset_loader.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("bert-base-uncased")

    print("=" * 60)
    print("Testing Stage 1 DataLoader (NLI only, 100 samples) …")
    print("=" * 60)

    nli_pairs = load_nli_pairs(max_samples=100)
    ds  = PairDataset(nli_pairs, tok, max_length=128)
    dl  = DataLoader(ds, batch_size=8, collate_fn=ds.collate_fn)
    batch = next(iter(dl))

    print(f"\nBatch keys: {list(batch.keys())}")
    print(f"query_input_ids shape    : {batch['query_input_ids'].shape}")
    print(f"document_input_ids shape : {batch['document_input_ids'].shape}")

    print("\nSampling probabilities with α=0.5:")
    sizes = [100, 500, 1000]
    probs = compute_sampling_probabilities(sizes, alpha=0.5)
    for s, p in zip(sizes, probs):
        print(f"  n={s:5d}  →  p={p:.4f}")

    print("\n✓ dataset_loader.py sanity check complete.")
