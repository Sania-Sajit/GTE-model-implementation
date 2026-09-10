"""
bm25_miner.py
──────────────
BM25 Hard Negative Miner for Stage 2 Supervised Fine-Tuning (Section 3.2 of GTE paper).

Paper reference:
  "In Stage 2, for each query q_i and positive document d_i^+,
   we index candidate documents using BM25 and retrieve top-k
   ranking documents that do NOT contain the positive answer."

Uses rank_bm25 (BM25Okapi).
"""

from typing import List, Tuple
from rank_bm25 import BM25Okapi


def tokenize_text(text: str) -> List[str]:
    """Basic whitespace + lowercase tokenization."""
    return text.lower().strip().split()


class BM25HardNegativeMiner:
    """
    BM25 Hard Negative Indexer & Miner.
    """

    def __init__(self, corpus: List[str]):
        """
        Build a BM25 index over candidate document texts.
        """
        self.corpus = corpus
        self.tokenized_corpus = [tokenize_text(doc) for doc in corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def mine_hard_negatives(
        self,
        query: str,
        positive_doc: str,
        k: int = 7,
    ) -> List[str]:
        """
        Retrieve top-k BM25 documents for `query` that are NOT equal to `positive_doc`.
        """
        query_tokens = tokenize_text(query)
        # Get top-N candidates to ensure we have k non-positive documents
        top_candidates = self.bm25.get_top_n(query_tokens, self.corpus, n=k + 15)

        hard_negs: List[str] = []
        pos_clean = positive_doc.strip().lower()

        for cand in top_candidates:
            cand_clean = cand.strip().lower()
            if cand_clean != pos_clean and cand not in hard_negs:
                hard_negs.append(cand)
            if len(hard_negs) >= k:
                break

        # Fallback if corpus is small or no non-positive candidate found
        while len(hard_negs) < k:
            fallback = self.corpus[len(hard_negs) % len(self.corpus)]
            if fallback.strip().lower() != pos_clean:
                hard_negs.append(fallback)
            else:
                hard_negs.append(positive_doc)  # Safety fallback

        return hard_negs[:k]
