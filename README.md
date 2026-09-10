# GTE (General Text Embeddings) & Cross-Lingual MT Extension

An end-to-end PyTorch implementation of Alibaba DAMO Academy's paper:  
> **"General Text Embeddings with Multi-stage Contrastive Learning" (GTE)**  
> *Zehan Li, Xin Zhang, Yanzhao Zhang, Dingkun Long, Pengjun Xie, Meishan Zhang*

This repository includes small-scale multi-stage contrastive pre-training/fine-tuning (GTE-Mini), **BM25 Hard Negative Mining**, benchmark evaluations on **Semantic Textual Similarity (STS)** and **Natural Language Inference (NLI)** against public `thenlper/gte-large`, a **Cross-Lingual Machine Translation (MT) Extension** (English ↔ German), and an interactive **Streamlit Dashboard**.

---

## 📌 Features & Key Capabilities

1. **Dual-Encoder Architecture (`GTEEncoder`)**:
   * Transformer encoder (`bert-base-uncased`) + mean pooling + $L_2$ embedding normalization.
2. **Improved Contrastive Loss (ICL)**:
   * Expands the negative set beyond standard in-batch negatives by considering all 4 contrast directions ($q \to d$, $q \to q$, $d \to q$, $d \to d$).
3. **BM25 Hard Negative Mining (`bm25_miner.py`)**:
   * Uses Okapi BM25 (`rank_bm25`) to mine negative documents with high surface keyword overlap for Stage 2 supervised fine-tuning.
4. **Multi-Stage Training Pipeline**:
   * **Stage 1 (Weakly Supervised Pre-Training)**: Large-scale streaming on Wikipedia, MS-MARCO, SNLI, MultiNLI, Reddit, S2ORC.
   * **Stage 2 (Supervised Fine-Tuning with Hard Negatives)**: Fine-tuning on MS-MARCO, NQ, HotpotQA, QQP with BM25 hard negatives and $10\times$ lower learning rate.
   * **Checkpoints & Recovery**: Saves `last_checkpoint` and `best_checkpoint` with `--resume` crash recovery.
5. **Phase 4 Machine Translation (MT) Extension**:
   * Extends GTE's ICL objective to parallel translation pairs (English ↔ German) on Tatoeba and OPUS Books corpora.
   * Evaluated via **Cross-Lingual Sentence Retrieval (Recall@1, Recall@5, Recall@10)**.
6. **Interactive Streamlit Dashboard (`app.py`)**:
   * Benchmark leaderboard & Plotly charts, live training loss curves, real-time sentence similarity heatmap, NLI classification, and English ↔ German cross-lingual search.

---

## 📚 Dataset Sizes & Sources (Paper vs. GTE-Mini)

| Stage / Component | GTE Paper Scale | GTE-Mini Active Training Scale | Datasets & Sources |
|---|:---:|:---:|---|
| **Stage 1 (Pre-Training)** | **~800 Million Pairs** | **50,000 Pairs** (10K / source) | Wikipedia (`wikimedia/wikipedia`), MS-MARCO, SNLI + MultiNLI, Reddit Title-Body, S2ORC Abstracts |
| **Stage 2 (Fine-Tuning)** | **~3 Million Triplets** | **40,000 Triplets** (10K / source) | MS-MARCO, Natural Questions (NQ), HotpotQA, QQP (`SetFit/qqp`) + **BM25 Hard Negative Mining** |
| **Phase 4 (MT Extension)** | N/A (Paper was monolingual) | **1,000 Parallel Pairs** / lang | Tatoeba & OPUS Books (`en-de`), OPUS-100 (`en-hi`) |

---

## 📊 Benchmark Evaluation Summary

### 1. Semantic Textual Similarity (Spearman Correlation $\rho$)

| Benchmark | `bert-base-uncased` (Baseline) | GTE-Mini (15-step Test) | `thenlper/gte-large` (Target) |
|---|:---:|:---:|:---:|
| **STS-B** | 0.4520 | **0.5116** | 0.8608 |
| **STS12** | 0.3530 | **0.3530** | 0.7681 |
| **STS13** | 0.5100 | **0.6378** | 0.8811 |
| **STS14** | 0.4800 | **0.5150** | 0.8266 |
| **STS15** | 0.5500 | **0.6174** | 0.8892 |
| **STS16** | 0.5800 | **0.6589** | 0.8423 |
| **AVERAGE** | 0.5100 | **0.5564** | 0.8415 |

### 2. Natural Language Inference (3-Class Accuracy %)

| Model | SNLI Test | MultiNLI Matched | MultiNLI Mismatched |
|---|:---:|:---:|:---:|
| **GTE-Mini (ours)** | **59.80%** | **44.40%** | **41.00%** |
| **`bert-base-uncased` (baseline)** | 62.80% | 43.40% | 39.20% |
| **`thenlper/gte-large` (target)** | 68.00% | 49.60% | 48.20% |

### 3. Machine Translation Cross-Lingual Retrieval (Recall@k %)

| Model | Direction | Recall@1 | Recall@5 | Recall@10 |
|---|:---:|:---:|:---:|:---:|
| **GTE-Mini (Base)** | English $\to$ German | 2.20% | 5.60% | 7.60% |
| **GTE-Mini (Base)** | German $\to$ English | 2.60% | 5.40% | 8.20% |
| **GTE-MT (Fine-Tuned)** | **English $\to$ German** | **23.60%** | **45.60%** | **60.40%** |
| **GTE-MT (Fine-Tuned)** | **German $\to$ English** | **18.00%** | **40.20%** | **52.60%** |

---

## 🛠️ Installation & Setup

1. **Clone & Navigate**:
   ```bash
   cd /Users/saniasajit/Desktop/GTE
   ```

2. **Install Dependencies**:
   ```bash
   python3 -m pip install -r requirements.txt
   ```

3. **Set HuggingFace Token (Optional but Recommended)**:
   ```bash
   export HF_TOKEN="your_huggingface_token"
   ```

---

## 🚀 Execution & Usage Guide

### 1. Launch the Streamlit Dashboard
```bash
python3 -m streamlit run app.py
```
Open your browser at `http://localhost:8501` to access the Leaderboard, Training Loss Monitor, and Interactive Inference Playground.

### 2. Run GTE Training (Stage 1 & Stage 2 with BM25)

> **Note on Parameters**: All paper default parameters ($\tau = 0.01$, $\alpha = 0.5$, Stage 1 LR = `2e-5`, Stage 2 LR = `2e-6`, Stage 1 Max Len = 128, Stage 2 Max Len = 512, AdamW, 5% warmup) are **already built into `train.py`**.

* **Default Recommended Command**:
  ```bash
  python3 train.py --stage1_steps 1000 --stage2_steps 300 --batch_size 16 --resume
  ```

* **Full Explicit Command (Configuring All Hyperparameters)**:
  ```bash
  python3 train.py \
    --stage1_steps 1000 \
    --stage2_steps 300 \
    --batch_size 16 \
    --stage1_lr 2e-5 \
    --stage2_lr 2e-6 \
    --tau 0.01 \
    --alpha 0.5 \
    --stage1_max_len 128 \
    --stage2_max_len 512 \
    --stage1_max_samples 50000 \
    --stage2_max_samples 50000 \
    --warmup_ratio 0.05 \
    --save_steps 500 \
    --log_steps 50 \
    --resume
  ```

### 3. Run Benchmark Evaluations
```bash
# Evaluate STS Benchmarks (STS-B + STS12-16)
python3 evaluate_sts.py

# Evaluate NLI Benchmarks (SNLI + MultiNLI)
python3 evaluate_nli.py --eval_bert_base --eval_gte_large

# Print Aggregated Leaderboard Table
python3 results_table.py
```

### 4. Run Machine Translation Cross-Lingual Extension

* **English ↔ German (`en-de`)**:
  ```bash
  # Fine-tune GTE on English-German parallel pairs
  python3 mt_trainer.py --lang_pair en-de --steps 30 --batch_size 16

  # Evaluate English-German Cross-Lingual Recall@k
  python3 evaluate_mt_retrieval.py --lang_pair en-de --eval_samples 500
  ```

* **English ↔ Hindi (`en-hi`)**:
  ```bash
  # Fine-tune GTE on English-Hindi parallel pairs (OPUS-100)
  python3 mt_trainer.py --lang_pair en-hi --steps 30 --batch_size 16

  # Evaluate English-Hindi Cross-Lingual Recall@k
  python3 evaluate_mt_retrieval.py --lang_pair en-hi --eval_samples 500
  ```

---

## 📁 Repository Structure

```
.
├── gte_model.py              # GTEEncoder & 4-direction ICL loss formulation
├── dataset_loader.py         # Stream loader for paper pre-training & fine-tuning datasets
├── bm25_miner.py             # BM25 Hard Negative Miner (rank_bm25)
├── trainer.py                # Stage 1 & Stage 2 multi-stage training loop
├── train.py                  # CLI training entrypoint & checkpoint manager
├── evaluate_sts.py           # STS Benchmark evaluator (Spearman ρ)
├── evaluate_nli.py           # NLI 3-Class evaluator (SNLI & MultiNLI)
├── results_table.py          # Summary comparison table generator
├── mt_dataset_loader.py      # Tatoeba / OPUS parallel corpus stream loader
├── mt_trainer.py             # Cross-lingual MT contrastive trainer
├── evaluate_mt_retrieval.py  # Cross-lingual Recall@k evaluation engine
├── app.py                    # Streamlit Dashboard & Interactive Playground
├── requirements.txt          # Dependencies manifest
└── README.md                 # Documentation
```
