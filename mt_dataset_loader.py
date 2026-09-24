"""
mt_dataset_loader.py
────────────────────
DataLoader for Machine Translation (MT) parallel sentence pairs.
Supports English ↔ German (en-de) and English ↔ Hindi (en-hi).

Dataset sources:
  - Helsinki-NLP/opus_books (config: 'de-en')
  - Helsinki-NLP/opus-100 (config: 'en-hi')
  - wmt/wmt19 (config: 'de-en')
"""

import sys
from typing import List, Optional, NamedTuple
from torch.utils.data import Dataset, DataLoader
import torch


class MTPair(NamedTuple):
    src_text: str  # English or target language sentence
    tgt_text: str  # Corresponding translation in target language


def load_opus_de_en_pairs(max_samples: Optional[int] = None, split: str = "train") -> List[MTPair]:
    """Load parallel sentences from OPUS Books (de-en)."""
    from datasets import load_dataset

    print(f"Loading OPUS Books (de-en, split='{split}') parallel corpus …")
    pairs: List[MTPair] = []
    
    # Try OPUS books first, fallback to OPUS-100 de-en if needed
    for repo_name, config_name in [("Helsinki-NLP/opus_books", "de-en"), ("Helsinki-NLP/opus-100", "de-en")]:
        try:
            ds = load_dataset(repo_name, config_name, split=split, streaming=True)
            for ex in ds:
                tr = ex.get("translation", {})
                de = (tr.get("de") or "").strip()
                en = (tr.get("en") or "").strip()
                if de and en:
                    pairs.append(MTPair(src_text=en, tgt_text=de))
                if max_samples and len(pairs) >= max_samples:
                    break
            if pairs:
                break
        except Exception as e:
            print(f"WARNING: {repo_name} split='{split}' failed ({e}). Trying fallback…")

    print(f"Loaded {len(pairs):,} parallel pairs (en ↔ de, split='{split}').")
    return pairs


def load_opus_hi_en_pairs(max_samples: Optional[int] = None, split: str = "train") -> List[MTPair]:
    """Load parallel sentences for English ↔ Hindi from OPUS-100 (en-hi)."""
    from datasets import load_dataset

    print(f"Loading OPUS-100 (en-hi, split='{split}') parallel corpus …")
    pairs: List[MTPair] = []
    try:
        ds = load_dataset("Helsinki-NLP/opus-100", "en-hi", split=split, streaming=True)
        for ex in ds:
            tr = ex.get("translation", {})
            hi = (tr.get("hi") or "").strip()
            en = (tr.get("en") or "").strip()
            if hi and en:
                pairs.append(MTPair(src_text=en, tgt_text=hi))
            if max_samples and len(pairs) >= max_samples:
                break
    except Exception as e:
        print(f"WARNING: OPUS-100 en-hi split='{split}' failed to load ({e}).")

    print(f"Loaded {len(pairs):,} parallel pairs (OPUS-100 en ↔ hi, split='{split}').")
    return pairs


def load_tatoeba_hi_en_pairs(max_samples: Optional[int] = None, split: str = "test") -> List[MTPair]:
    """Load high-quality parallel sentences for English ↔ Hindi from Tatoeba (mteb/tatoeba-bitext-mining hin-eng)."""
    from datasets import load_dataset

    print(f"Loading Tatoeba (hin-eng, split='{split}') parallel corpus …")
    pairs: List[MTPair] = []
    try:
        ds = load_dataset("mteb/tatoeba-bitext-mining", "hin-eng", split=split, streaming=True)
        for ex in ds:
            hi = (ex.get("sentence1") or "").strip()
            en = (ex.get("sentence2") or "").strip()
            if hi and en:
                pairs.append(MTPair(src_text=en, tgt_text=hi))
            if max_samples and len(pairs) >= max_samples:
                break
    except Exception as e:
        print(f"WARNING: Tatoeba hin-eng split='{split}' failed to load ({e}).")

    print(f"Loaded {len(pairs):,} parallel pairs (Tatoeba en ↔ hi, split='{split}').")
    return pairs


def load_tatoeba_de_en_pairs(max_samples: Optional[int] = None, split: str = "test") -> List[MTPair]:
    """Load high-quality parallel sentences for English ↔ German from Tatoeba (mteb/tatoeba-bitext-mining deu-eng)."""
    from datasets import load_dataset

    print(f"Loading Tatoeba (deu-eng, split='{split}') parallel corpus …")
    pairs: List[MTPair] = []
    try:
        ds = load_dataset("mteb/tatoeba-bitext-mining", "deu-eng", split=split, streaming=True)
        for ex in ds:
            de = (ex.get("sentence1") or "").strip()
            en = (ex.get("sentence2") or "").strip()
            if de and en:
                pairs.append(MTPair(src_text=en, tgt_text=de))
            if max_samples and len(pairs) >= max_samples:
                break
    except Exception as e:
        print(f"WARNING: Tatoeba deu-eng split='{split}' failed to load ({e}).")

    print(f"Loaded {len(pairs):,} parallel pairs (Tatoeba en ↔ de, split='{split}').")
    return pairs


def load_tatoeba_pairs(max_samples: Optional[int] = None, lang_pair: str = "en-de", split: str = "train", dataset_name: str = "auto") -> List[MTPair]:
    """
    Load parallel sentences from Tatoeba / OPUS corpora.
    Supports lang_pair='en-de' or 'en-hi', dataset_name='opus100' or 'tatoeba', and split='train' or 'test'.
    """
    if lang_pair == "en-hi":
        if dataset_name == "tatoeba":
            return load_tatoeba_hi_en_pairs(max_samples=max_samples, split=split)
        else:
            return load_opus_hi_en_pairs(max_samples=max_samples, split=split)
    else:
        if dataset_name == "tatoeba":
            return load_tatoeba_de_en_pairs(max_samples=max_samples, split=split)
        else:
            return load_opus_de_en_pairs(max_samples=max_samples, split=split)



class MTPairDataset(Dataset):
    def __init__(self, pairs: List[MTPair]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> MTPair:
        return self.pairs[idx]


def build_mt_dataloader(
    tokenizer,
    max_samples: Optional[int] = None,
    batch_size: int = 16,
    max_length: int = 128,
    lang_pair: str = "en-de",
    split: str = "train",
    dataset_name: str = "auto",
) -> DataLoader:
    """Build PyTorch DataLoader for MT training / evaluation."""
    pairs = load_tatoeba_pairs(max_samples=max_samples, lang_pair=lang_pair, split=split, dataset_name=dataset_name)


    if not pairs:
        print(f"ERROR: No MT parallel pairs could be loaded for language pair '{lang_pair}'.")
        sys.exit(1)

    dataset = MTPairDataset(pairs)

    def collate_fn(batch: List[MTPair]):
        src_texts = [b.src_text for b in batch]
        tgt_texts = [b.tgt_text for b in batch]

        src_tok = tokenizer(
            src_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tgt_tok = tokenizer(
            tgt_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        return {
            "query_input_ids": src_tok["input_ids"],
            "query_attention_mask": src_tok["attention_mask"],
            "doc_input_ids": tgt_tok["input_ids"],
            "doc_attention_mask": tgt_tok["attention_mask"],
        }

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
    )
