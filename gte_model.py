"""
gte_model.py
────────────
Core GTE architecture:
  - GTEEncoder  : Transformer (bert-base-uncased) + mean pooling → sentence embedding
  - improved_contrastive_loss : GTE's ICL objective (Section 3.3 of paper)
      Expands the negative set beyond standard in-batch negatives by considering
      all four contrast directions: q→d, q→q, d→q, d→d

Paper reference:
  "General Text Embeddings with Multi-stage Contrastive Learning"
  (GTE), Alibaba DAMO Academy

Required Dependencies:
  pip install torch transformers datasets accelerate scikit-learn scipy tqdm requests
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


# ─────────────────────────────────────────────────────────────────────────────
# 1.  GTEEncoder
# ─────────────────────────────────────────────────────────────────────────────

class GTEEncoder(nn.Module):
    """
    Dual-encoder with shared weights.

    Architecture (as in the GTE paper):
        Text
         ↓
        Transformer encoder  (bert-base-uncased for GTE-base)
         ↓
        Contextualized token representations
         ↓
        Mean pooling  (over non-padding tokens)
         ↓
        Text embedding  ∈ ℝ^d

    The SAME encoder is used for both queries and documents (shared weights).
    Contrastive learning is NOT a layer inside the encoder — it is the training
    objective applied to the resulting embeddings (see improved_contrastive_loss).
    """

    def __init__(self, model_name: str = "bert-base-uncased"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size

    # ── Mean Pooling ──────────────────────────────────────────────────────────

    @staticmethod
    def mean_pool(
        token_embeddings: torch.Tensor,   # (B, L, D)
        attention_mask:   torch.Tensor,   # (B, L)
    ) -> torch.Tensor:                    # (B, D)
        """
        Average the token embeddings, ignoring padding tokens.

        Formula (from the paper):
            e = Σ_t (mask_t · h_t) / Σ_t mask_t

        where h_t is the token embedding at position t and mask_t ∈ {0,1}.
        """
        # Expand mask to match embedding dimension: (B, L) → (B, L, D)
        mask_expanded = attention_mask.unsqueeze(-1).float()

        # Weighted sum over non-padding tokens
        sum_embeddings = (token_embeddings * mask_expanded).sum(dim=1)   # (B, D)
        sum_mask       = mask_expanded.sum(dim=1).clamp(min=1e-9)        # (B, D)

        return sum_embeddings / sum_mask  # (B, D)

    # ── Forward pass ─────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids:      torch.Tensor,   # (B, L)
        attention_mask: torch.Tensor,   # (B, L)
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:                  # (B, D)
        """
        Encode a batch of texts and return their mean-pooled embeddings.
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # outputs.last_hidden_state : (B, L, D)
        embeddings = self.mean_pool(outputs.last_hidden_state, attention_mask)
        return embeddings

    # ── Convenience: encode raw strings ──────────────────────────────────────

    @torch.no_grad()
    def encode(
        self,
        texts: list[str],
        max_length: int = 512,
        batch_size: int = 64,
        device: str | torch.device = "cpu",
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Encode a list of strings into embeddings (no gradient).
        Used during evaluation.

        Args:
            texts       : list of raw strings
            max_length  : maximum token sequence length
            batch_size  : tokenisation / forward-pass batch size
            device      : target device
            normalize   : if True, L2-normalise before returning
                          (cosine similarity then = dot product)

        Returns:
            Tensor of shape (N, D)
        """
        self.eval()
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}

            embs = self(**encoded)          # (b, D)
            if normalize:
                embs = F.normalize(embs, p=2, dim=-1)
            all_embeddings.append(embs.cpu())

        return torch.cat(all_embeddings, dim=0)   # (N, D)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Cosine Similarity
# ─────────────────────────────────────────────────────────────────────────────

def cosine_similarity_matrix(
    a: torch.Tensor,   # (N, D)  — already L2-normalised
    b: torch.Tensor,   # (M, D)  — already L2-normalised
) -> torch.Tensor:     # (N, M)
    """
    Compute pairwise cosine similarity between every row in a and every row in b.

        s(q, d) = q · d / (‖q‖₂ · ‖d‖₂)

    Because we L2-normalise inputs, this simplifies to a matrix multiplication:
        S = A @ Bᵀ      (values in [-1, 1])
    """
    a = F.normalize(a, p=2, dim=-1)
    b = F.normalize(b, p=2, dim=-1)
    return a @ b.t()   # (N, M)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  GTE Improved Contrastive Loss  (ICL)
# ─────────────────────────────────────────────────────────────────────────────

def improved_contrastive_loss(
    q_embs: torch.Tensor,   # (N, D)  query embeddings    (NOT yet normalised)
    d_embs: torch.Tensor,   # (N, D)  document embeddings (NOT yet normalised)
    tau:    float = 0.01,   # temperature τ (paper sets τ = 0.01)
) -> torch.Tensor:           # scalar loss
    """
    GTE Improved Contrastive Learning (ICL) objective.

    Standard in-batch negatives (SimCLR / DPR style) only contrast:
        q_i  vs  {d_j : j ≠ i}

    GTE expands the negative set to all four directions (Section 3.3):
        1.  q_i → d_j   (query-to-document)          [asymmetric]
        2.  q_i → q_j   (query-to-query)              [symmetric among queries]
        3.  d_j → d_i   (document-to-document)        [symmetric among docs]
        4.  d_j → q_i   (inverse: document-to-query)  [asymmetric]

    This gives more negative comparisons from the SAME batch, improving
    the contrastive signal without needing a larger batch.

    The ICL loss for a single query q_i:

        L_i = -log  exp(s(q_i, d_i) / τ)
                    ─────────────────────────────────────────────────
                    Σ_{j≠i} exp(s(q_i, d_j)/τ)  +  Σ_{j≠i} exp(s(q_i, q_j)/τ)
                    +  exp(s(q_i, d_i)/τ)          ← positive in denominator too

    And symmetrically for d_i.  The full loss averages over all i.

    Args:
        q_embs  : (N, D) query   embeddings (raw, before normalisation)
        d_embs  : (N, D) document embeddings (raw, before normalisation)
        tau     : temperature (paper uses 0.01)

    Returns:
        scalar loss value
    """
    N = q_embs.size(0)

    # ── L2 normalise ─────────────────────────────────────────────────────────
    q = F.normalize(q_embs, p=2, dim=-1)   # (N, D)
    d = F.normalize(d_embs, p=2, dim=-1)   # (N, D)

    # ── All pairwise similarities ─────────────────────────────────────────────
    # Scale by 1/τ up-front so the softmax is numerically equivalent
    s_qq = (q @ q.t()) / tau   # (N, N)  — query-to-query
    s_qd = (q @ d.t()) / tau   # (N, N)  — query-to-document
    s_dq = (d @ q.t()) / tau   # (N, N)  — document-to-query
    s_dd = (d @ d.t()) / tau   # (N, N)  — document-to-document

    # Positive pair indices: diagonal  (q_i, d_i)
    labels = torch.arange(N, device=q_embs.device)   # [0, 1, …, N-1]

    # ── Query-side loss ───────────────────────────────────────────────────────
    # Numerator for q_i: s(q_i, d_i)
    # Denominator: all q→d similarities  +  all q→q similarities (excluding self)
    #
    # To exclude the self-similarity term q_i → q_i from the denominator,
    # we mask the diagonal of s_qq with -inf before concatenating.
    s_qq_masked = s_qq.clone()
    s_qq_masked.fill_diagonal_(float("-inf"))

    # Concatenate along the "candidate" dimension: [q→d | q→q\{self}]
    logits_q = torch.cat([s_qd, s_qq_masked], dim=1)   # (N, 2N)
    # Positive index in this concatenated view is still `labels` (column i of s_qd)
    loss_q = F.cross_entropy(logits_q, labels)

    # ── Document-side loss (symmetric) ───────────────────────────────────────
    s_dd_masked = s_dd.clone()
    s_dd_masked.fill_diagonal_(float("-inf"))

    logits_d = torch.cat([s_dq, s_dd_masked], dim=1)   # (N, 2N)
    loss_d = F.cross_entropy(logits_d, labels)

    # ── Average the two sides ─────────────────────────────────────────────────
    return (loss_q + loss_d) / 2.0


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Quick sanity check  (run with:  python gte_model.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import math

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running sanity check on device: {device}\n")

    # ── 4a. Model loads ───────────────────────────────────────────────────────
    print("Loading GTEEncoder (bert-base-uncased) …")
    model = GTEEncoder("bert-base-uncased").to(device)
    print(f"  Hidden size : {model.hidden_size}")
    print(f"  Parameters  : {sum(p.numel() for p in model.parameters()):,}\n")

    # ── 4b. Forward pass ──────────────────────────────────────────────────────
    queries   = ["What is machine learning?",
                 "How do I reset my password?"]
    documents = ["Machine learning is a branch of AI.",
                 "Steps to recover your account credentials."]

    q_embs = model.encode(queries,   device=device, normalize=False)
    d_embs = model.encode(documents, device=device, normalize=False)

    print(f"Query embedding shape    : {q_embs.shape}")     # (2, 768)
    print(f"Document embedding shape : {d_embs.shape}\n")   # (2, 768)

    # ── 4c. Cosine similarity ─────────────────────────────────────────────────
    sim = cosine_similarity_matrix(q_embs, d_embs)
    print("Cosine similarity matrix (queries × documents):")
    print(sim)
    print()

    # ── 4d. ICL loss ──────────────────────────────────────────────────────────
    # Move to device for loss computation
    q_embs = q_embs.to(device)
    d_embs = d_embs.to(device)
    loss = improved_contrastive_loss(q_embs, d_embs, tau=0.01)
    print(f"ICL loss (N=2, τ=0.01): {loss.item():.4f}")

    # ── 4e. Sanity: diagonal should be highest in sim matrix ──────────────────
    diag_avg = sim.diag().mean().item()
    off_diag = (sim.sum() - sim.diag().sum()) / (sim.numel() - sim.size(0))
    print(f"\nSanity check (BERT is not yet trained — diag ≈ off-diag):")
    print(f"  Mean diagonal similarity     : {diag_avg:.4f}")
    print(f"  Mean off-diagonal similarity : {off_diag.item():.4f}")
    print("\n✓ gte_model.py sanity check complete.")
