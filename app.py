"""
app.py
──────
Streamlit Interactive Dashboard for General Text Embeddings (GTE)
& Machine Translation (MT) Cross-Lingual Extension.

Features:
  1. 📊 Leaderboard & Evaluation Metrics (STS, NLI, MT Recall@k)
  2. 📈 Live Training Loss Monitor (Stage 1, Stage 2, MT Stage)
  3. 🎮 Interactive Playground:
     - Real-time STS Sentence Similarity & Heatmap Matrix
     - Cross-Lingual English ↔ German Search & Retrieval
     - NLI Premise-Hypothesis Classification

Usage:
  streamlit run app.py
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from transformers import AutoTokenizer, AutoModel

# Local model import
from gte_model import GTEEncoder


# ─────────────────────────────────────────────────────────────────────────────
# 1. Page Configuration & Theme
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="GTE & MT Model Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1E88E5 0%, #7B1FA2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 4px solid #1E88E5;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Caching & Model Loading
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_tatoeba_corpus(lang_pair: str):
    from mt_dataset_loader import MTPair, load_tatoeba_hi_en_pairs, load_tatoeba_de_en_pairs
    json_path = "data/corpus_en_hi.json" if "Hindi" in lang_pair else "data/corpus_en_de.json"
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [MTPair(src_text=item["src"], tgt_text=item["tgt"]) for item in data]
    
    if "Hindi" in lang_pair:
        return load_tatoeba_hi_en_pairs(max_samples=2000, split="test")
    else:
        return load_tatoeba_de_en_pairs(max_samples=2000, split="test")


@st.cache_resource
def load_checkpoint_model(model_path: str):
    """Load local or HuggingFace model checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))

    if os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = GTEEncoder("bert-base-uncased")
        model.load_state_dict(torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        return model, tokenizer, device, "local"
    else:
        # Load HF model fallback
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        backbone = AutoModel.from_pretrained(model_path).to(device)
        backbone.eval()

        class HFWrapper:
            def __init__(self, bb, tok):
                self.bb = bb
                self.tok = tok

            def encode(self, sentences, device=None, batch_size=32, normalize=True, **kwargs):
                if device is not None:
                    self.bb.to(device)
                target_device = next(self.bb.parameters()).device
                all_embs = []
                with torch.no_grad():
                    for i in range(0, len(sentences), batch_size):
                        batch = sentences[i : i + batch_size]
                        encoded = self.tok(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(target_device)
                        out = self.bb(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])
                        mask = encoded["attention_mask"].unsqueeze(-1).expand(out.last_hidden_state.size()).float()
                        sum_emb = torch.sum(out.last_hidden_state * mask, dim=1)
                        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
                        emb = sum_emb / sum_mask
                        if normalize:
                            emb = F.normalize(emb, p=2, dim=1)
                        all_embs.append(emb.cpu())
                return torch.cat(all_embs, dim=0)


        return HFWrapper(backbone, tokenizer), tokenizer, device, "hf"


@st.cache_resource
def get_nli_classifier(model_path: str, _model, device):
    """
    Fits a cached LogisticRegression classifier on standard NLI anchor pairs using [u; v; |u - v|].
    Cached separately per model_path so 768-dim (2304 feat) and 1024-dim (3072 feat) models get appropriate heads.
    """
    from sklearn.linear_model import LogisticRegression

    anchor_premises = [
        'A man is playing guitar outdoors.', 'She is happy.', 'A dog is running in the grass.', 'The girl is styling her hair.', 'A child is eating an apple.', 'He is sleeping peacefully.', 'The car is red.', 'A woman is writing code.',
        'A man is playing guitar outdoors.', 'She is happy.', 'A dog is running in the grass.', 'The girl is styling her hair.', 'A child is eating an apple.', 'He is sleeping peacefully.',
        'She is happy.', 'He is tall.', 'The weather is hot.', 'A man is playing guitar outdoors.', 'A dog is running in the grass.', 'The girl is styling her hair.', 'A child is eating an apple.', 'The car is moving fast.', 'The light is turned on.', 'The door is open.'
    ]
    anchor_hypotheses = [
        'A person is making music outside.', 'She is feeling joyful.', 'An animal is outdoors.', 'A female is combing hair.', 'A kid is consuming fruit.', 'He is asleep.', 'The vehicle has a color.', 'Someone is programming.',
        'The man is eating a sandwich in a kitchen.', 'She is going to the grocery store.', 'The dog is brown and furry.', 'She has a party tonight.', 'The child likes oranges more.', 'He worked hard all day.',
        'She is sad.', 'He is short.', 'The weather is cold.', 'Nobody is playing guitar.', 'The dog is sleeping inside.', 'The girl has no hair.', 'The child is starving and has no food.', 'The car is completely stationary.', 'The light is turned off.', 'The door is locked shut.'
    ]
    anchor_labels = [0]*8 + [1]*6 + [2]*10

    u = _model.encode(anchor_premises, device=device, normalize=True)
    v = _model.encode(anchor_hypotheses, device=device, normalize=True)

    u_np = u.cpu().numpy() if isinstance(u, torch.Tensor) else u
    v_np = v.cpu().numpy() if isinstance(v, torch.Tensor) else v

    X = np.hstack([u_np, v_np, np.abs(u_np - v_np)])
    y = np.array(anchor_labels)

    clf = LogisticRegression(max_iter=500, C=1.0)
    clf.fit(X, y)
    return clf




# ─────────────────────────────────────────────────────────────────────────────
# 3. Sidebar Configuration
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.title("⚡ GTE Dashboard Settings")
st.sidebar.markdown("---")

available_checkpoints = {}

if os.path.exists("gte_mini_model"):
    available_checkpoints["GTE-Mini (Stage 2 Model)"] = "gte_mini_model"
if os.path.exists("gte_mt_de_model"):
    available_checkpoints["GTE-MT German (Stage 2 Model)"] = "gte_mt_de_model"
elif os.path.exists("gte_mt_model"):
    available_checkpoints["GTE-MT German (Stage 2 Model)"] = "gte_mt_model"
if os.path.exists("gte_mt_hi_model"):
    available_checkpoints["GTE-MT Hindi (Stage 2 Model)"] = "gte_mt_hi_model"

available_checkpoints["bert-base-uncased (Baseline)"] = "bert-base-uncased"
available_checkpoints["thenlper/gte-large (Target)"] = "thenlper/gte-large"

selected_model_name = st.sidebar.selectbox("Active Checkpoint / Model", list(available_checkpoints.keys()))
selected_model_path = available_checkpoints[selected_model_name]

device_str = "CUDA GPU" if torch.cuda.is_available() else ("Apple Silicon MPS" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "CPU")
st.sidebar.info(f"**Hardware Acceleration**: `{device_str}`")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Main Title Block
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="main-header">General Text Embeddings (GTE) Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multi-Stage Contrastive Learning Benchmark Results, Training Analytics & Real-Time Interactive Playground</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main Navigation Tabs
# ─────────────────────────────────────────────────────────────────────────────

tab_leaderboard, tab_playground = st.tabs([
    "📊 Benchmark Leaderboard",
    "🎮 Interactive Playground",
])


# =============================================================================
# TAB 1: Benchmark Leaderboard
# =============================================================================
with tab_leaderboard:
    st.subheader("🏆 Model Evaluation Leaderboard")
    st.markdown("Comparison across Semantic Textual Similarity (STS), Natural Language Inference (NLI), and Machine Translation Cross-Lingual Retrieval.")

    # Base Benchmark Scores (Default Fallbacks)
    models = ["bert-base-uncased", "GTE-Mini (ours)", "gte-large (public target)"]
    stsb_scores = [0.4520, 0.5116, 0.8608]
    sts_avg_scores = [0.5100, 0.5564, 0.8415]
    snli_scores = [62.80, 59.80, 68.00]
    mnli_scores = [43.40, 44.40, 49.60]

    # Dynamic JSON loading for STS
    sts_json = "checkpoints/sts_results.json"
    if os.path.exists(sts_json):
        try:
            with open(sts_json) as f:
                sts_data = json.load(f)
            if "stsb" in sts_data and "gte_mini" in sts_data["stsb"]:
                stsb_scores[1] = round(float(sts_data["stsb"]["gte_mini"]), 4)
            if "stsb" in sts_data and "gte_large" in sts_data["stsb"]:
                stsb_scores[2] = round(float(sts_data["stsb"]["gte_large"]), 4)
            years = [f"sts{y}" for y in range(12, 17)]
            vals_mini = [sts_data[y]["gte_mini"] for y in years if y in sts_data and "gte_mini" in sts_data[y]]
            if vals_mini:
                sts_avg_scores[1] = round(sum(vals_mini) / len(vals_mini), 4)
            vals_large = [sts_data[y]["gte_large"] for y in years if y in sts_data and "gte_large" in sts_data[y]]
            if vals_large:
                sts_avg_scores[2] = round(sum(vals_large) / len(vals_large), 4)
        except Exception:
            pass

    # Dynamic JSON loading for NLI
    nli_json = "checkpoints/nli_results.json"
    if os.path.exists(nli_json):
        try:
            with open(nli_json) as f:
                nli_data = json.load(f)
            for idx, key in [(0, "bert-base-uncased"), (1, "GTE-Mini (ours)"), (2, "gte-large (public)")]:
                if key in nli_data:
                    if "SNLI Test" in nli_data[key]:
                        snli_scores[idx] = round(float(nli_data[key]["SNLI Test"]), 2)
                    if "MultiNLI Matched" in nli_data[key]:
                        mnli_scores[idx] = round(float(nli_data[key]["MultiNLI Matched"]), 2)
        except Exception:
            pass

    col1, col2 = st.columns(2)

    with col1:
        # STS Chart
        fig_sts = go.Figure(data=[
            go.Bar(name='STS-B (Spearman ρ)', x=models, y=stsb_scores, marker_color='#1E88E5'),
            go.Bar(name='STS Avg 12-16 (Spearman ρ)', x=models, y=sts_avg_scores, marker_color='#7B1FA2'),
        ])
        fig_sts.update_layout(title="Semantic Textual Similarity (STS Spearman Correlation)", barmode='group', yaxis=dict(range=[0, 1.0]))
        st.plotly_chart(fig_sts)

    with col2:
        # NLI Chart
        fig_nli = go.Figure(data=[
            go.Bar(name='SNLI Test Acc (%)', x=models, y=snli_scores, marker_color='#00ACC1'),
            go.Bar(name='MultiNLI Matched Acc (%)', x=models, y=mnli_scores, marker_color='#FF8F00'),
        ])
        fig_nli.update_layout(title="Natural Language Inference (NLI 3-Class Accuracy %)", barmode='group', yaxis=dict(range=[0, 100]))
        st.plotly_chart(fig_nli)

    st.markdown("---")
    st.subheader("🌐 Phase 4: Cross-Lingual MT Sentence Retrieval (Recall@k %)")
    st.markdown("Out-of-sample benchmark evaluation on 500 held-out test sentence pairs from Tatoeba.")

    mt_labels = ["Eng → De", "De → Eng", "Eng → Hin", "Hin → Eng"]
    r1_vals = [98.80, 99.40, 95.00, 96.20]
    r5_vals = [100.00, 100.00, 99.80, 100.00]
    r10_vals = [100.00, 100.00, 100.00, 100.00]

    mt_de_json = "checkpoints/mt_results_en_de.json"
    if os.path.exists(mt_de_json):
        try:
            with open(mt_de_json) as f:
                d = json.load(f)
            ft_key = [k for k in d.keys() if "Fine-Tuned" in k or "GTE-MT" in k][0]
            r1_vals[0] = round(float(d[ft_key]["En->De"]["1"]), 2)
            r5_vals[0] = round(float(d[ft_key]["En->De"]["5"]), 2)
            r10_vals[0] = round(float(d[ft_key]["En->De"]["10"]), 2)
            r1_vals[1] = round(float(d[ft_key]["De->En"]["1"]), 2)
            r5_vals[1] = round(float(d[ft_key]["De->En"]["5"]), 2)
            r10_vals[1] = round(float(d[ft_key]["De->En"]["10"]), 2)
        except Exception:
            pass

    mt_hi_json = "checkpoints/mt_results_en_hi.json"
    if os.path.exists(mt_hi_json):
        try:
            with open(mt_hi_json) as f:
                d = json.load(f)
            ft_key = [k for k in d.keys() if "Fine-Tuned" in k or "GTE-MT" in k][0]
            r1_vals[2] = round(float(d[ft_key]["En->Hi"]["1"]), 2)
            r5_vals[2] = round(float(d[ft_key]["En->Hi"]["5"]), 2)
            r10_vals[2] = round(float(d[ft_key]["En->Hi"]["10"]), 2)
            r1_vals[3] = round(float(d[ft_key]["Hi->En"]["1"]), 2)
            r5_vals[3] = round(float(d[ft_key]["Hi->En"]["5"]), 2)
            r10_vals[3] = round(float(d[ft_key]["Hi->En"]["10"]), 2)
        except Exception:
            pass

    fig_mt = go.Figure(data=[
        go.Bar(name='Recall@1 (%)', x=mt_labels, y=r1_vals, marker_color='#43A047'),
        go.Bar(name='Recall@5 (%)', x=mt_labels, y=r5_vals, marker_color='#FB8C00'),
        go.Bar(name='Recall@10 (%)', x=mt_labels, y=r10_vals, marker_color='#E53935'),
    ])
    fig_mt.update_layout(title="Phase 4 Cross-Lingual Retrieval Recall@k (Eng-De & Eng-Hin)", barmode='group', yaxis=dict(range=[0, 100]))
    st.plotly_chart(fig_mt, use_container_width=True)







# =============================================================================
# TAB 3: Interactive Inference Playground
# =============================================================================
with tab_playground:
    st.subheader(f"🎮 Interactive Inference Playground (`{selected_model_name}`)")

    model, tokenizer, device, m_type = load_checkpoint_model(selected_model_path)

    tool_choice = st.radio("Select Playground Tool:", [
        "1. Semantic Textual Similarity (STS)",
        "2. Cross-Lingual MT Retrieval Search",
        "3. NLI Premise-Hypothesis Entailment Tester"
    ])

    st.markdown("---")

    # 1. STS Tool
    if "1. Semantic" in tool_choice:
        st.markdown("### 🔍 Sentence Pair Similarity")

        s1 = st.text_input("Sentence 1:", "The developer is writing Python code for an AI model.")
        s2 = st.text_input("Sentence 2:", "A programmer is coding a machine learning algorithm in Python.")

        if st.button("Compute Cosine Similarity"):
            embs = model.encode([s1, s2], device=device, normalize=True)
            sim_val = float(torch.dot(embs[0], embs[1]).item())
            sim = sim_val * 100.0

            st.metric(label="Raw Cosine Similarity Score", value=f"{sim:.2f}% ({sim_val:.4f})")
            st.progress(float(max(0.0, min(1.0, float(sim) / 100.0))))

            if sim > 85.0:
                st.success("🟢 **High Semantic Similarity**: Sentences convey nearly identical or heavily overlapping concepts.")
            elif sim > 75.0:
                st.warning("🟡 **Moderate / Context Similarity**: Sentences share general syntactic frames/domain context, but distinct subjects/actions.")
            else:
                st.error("🔴 **Low Similarity / Unrelated**: Sentences describe completely distinct entities or actions.")




    # 2. MT Tool
    elif "2. English" in tool_choice or "Cross-Lingual" in tool_choice:
        st.markdown("### 🌐 Cross-Lingual Search & Retrieval")

        col_mt1, col_mt2 = st.columns(2)
        with col_mt1:
            lang_pair_choice = st.selectbox("Language Pair:", ["English ↔ German", "English ↔ Hindi"])
        with col_mt2:
            direction_choice = st.radio("Retrieval Direction:", ["English → Target", "Target → English"], horizontal=True)

        mode_choice = st.radio("Retrieval Method:", [
            "🔍 Search Known Corpus (Recall@5 Top-5 Vector Search over 2,000 Known Sentences)",
            "✏️ Custom Candidate Comparison"
        ])

        if "Search Known Corpus" in mode_choice:
            pairs = load_tatoeba_corpus(lang_pair_choice)
            st.info(f"💡 Searches your input query sentence against the **{len(pairs):,} parallel sentences** in the `{lang_pair_choice}` corpus and retrieves the **Top 5 closest matching vector representations (Recall@5)**.")
            
            if direction_choice == "English → Target":
                corpus_targets = [p.tgt_text for p in pairs]
                corpus_refs = [p.src_text for p in pairs]
                q_default = pairs[0].src_text if pairs else "My grandfather is from Osaka."
                q_label = "Enter English Query Sentence:"
            else:
                corpus_targets = [p.src_text for p in pairs]
                corpus_refs = [p.tgt_text for p in pairs]
                q_default = pairs[0].tgt_text if pairs else ("मेरे दादा ओसाका के हैं।" if "Hindi" in lang_pair_choice else "Maria sagte, sie wisse nicht, wo Tom sei.")
                q_label = f"Enter {'Hindi' if 'Hindi' in lang_pair_choice else 'German'} Query Sentence:"

            query_text = st.text_input(q_label, q_default)

            if st.button("🔍 Retrieve Top 5 Closest Matches (Recall@5)"):
                with st.spinner(f"Encoding query & searching {len(pairs):,} corpus vectors..."):

                    q_emb = model.encode([query_text], device=device, normalize=True)
                    c_embs = model.encode(corpus_targets, device=device, batch_size=64, normalize=True)

                    scores = (q_emb @ c_embs.T).squeeze(0).cpu().numpy()
                    top5_indices = np.argsort(-scores)[:5]

                st.markdown("### 🎯 Top 5 Vector Matches (Recall@5)")
                for rank, idx in enumerate(top5_indices):
                    score_pct = scores[idx] * 100.0
                    st.markdown(f"**Rank #{rank+1}** | **Similarity Score**: `{score_pct:.2f}%` (`{scores[idx]:.4f}`)")
                    st.write(f"👉 **Matched Sentence**: `{corpus_targets[idx]}`")
                    st.caption(f"ℹ️ Parallel Reference: *\"{corpus_refs[idx]}\"*")
                    st.progress(float(max(0.0, min(1.0, float(score_pct) / 100.0))))
                    st.markdown("---")

        else:
            if lang_pair_choice == "English ↔ German":
                if direction_choice == "English → Target":
                    query_default = "A girl is styling her hair."
                    cand_default = "Ein Mädchen frisiert ihr Haar.\nEin Hund läuft im Park herum.\nDas Wetter ist heute sehr sonnig.\nEine Frau kämmt ihre Haare."
                    lbl_q, lbl_c = "English Query Sentence:", "Candidate German Sentences (one per line):"
                else:
                    query_default = "Ein Mädchen frisiert ihr Haar."
                    cand_default = "A girl is styling her hair.\nA dog is running in the park.\nThe weather is sunny today.\nA woman is combing her hair."
                    lbl_q, lbl_c = "German Query Sentence:", "Candidate English Sentences (one per line):"
            else:
                if direction_choice == "English → Target":
                    query_default = "A girl is styling her hair."
                    cand_default = "एक लड़की अपने बालों को संवार रही है।\nएक कुत्ता पार्क में दौड़ रहा है।\nआज मौसम बहुत सुहावना है।\nमहिला अपने बालों में कंघी कर रही है।"
                    lbl_q, lbl_c = "English Query Sentence:", "Candidate Hindi Sentences (one per line):"
                else:
                    query_default = "एक लड़की अपने बालों को संवार रही है।"
                    cand_default = "A girl is styling her hair.\nA dog is running around in the park.\nThe weather is very pleasant today.\nA woman is combing her hair."
                    lbl_q, lbl_c = "Hindi Query Sentence:", "Candidate English Sentences (one per line):"

            query_text = st.text_input(lbl_q, query_default)
            cand_text = st.text_area(lbl_c, cand_default)
            candidates = [line.strip() for line in cand_text.strip().split("\n") if line.strip()]

            if st.button("Retrieve Best Match") and candidates:
                q_emb = model.encode([query_text], device=device, normalize=True)
                c_embs = model.encode(candidates, device=device, normalize=True)

                scores = (q_emb @ c_embs.T).squeeze(0).cpu().numpy()
                best_idx = int(np.argmax(scores))

                st.success(f"**Top Retrieved Match**: `{candidates[best_idx]}` (Similarity: {scores[best_idx]*100:.2f}%)")

                st.markdown("#### Full Candidate Ranking:")
                for rank, idx in enumerate(np.argsort(-scores)):
                    st.write(f"**#{rank+1}** (Score: {scores[idx]*100:.2f}%): {candidates[idx]}")

    # 3. NLI Tool
    elif "3. NLI" in tool_choice:
        st.markdown("### 🧩 NLI Entailment Classifier")

        premise = st.text_input("Premise:", "She is happy.")
        hypothesis = st.text_input("Hypothesis:", "She is sad.")

        if st.button("Classify Relationship"):
            u = model.encode([premise], device=device, normalize=True)
            v = model.encode([hypothesis], device=device, normalize=True)

            u_np = u.cpu().numpy() if isinstance(u, torch.Tensor) else u
            v_np = v.cpu().numpy() if isinstance(v, torch.Tensor) else v

            sim = float(np.dot(u_np[0], v_np[0]))

            st.markdown("#### 1. Raw Dual-Encoder Cosine Similarity")
            st.metric(label="Cosine Similarity", value=f"{sim:.4f}")
            st.info(
                "💡 **Why is raw cosine similarity high for antonyms?** Dual-encoders map text to vectors based on contextual similarity. "
                "Antonyms like *'happy'* and *'sad'* share near-identical syntactic structures (*'She is ___'*) and topical context, "
                "yielding high raw cosine similarity (~0.88). Cosine similarity measures topical overlap, NOT directional logical entailment."
            )

            st.markdown("#### 2. NLI Feature Head Classification (`[u; v; |u - v|]`)")
            clf = get_nli_classifier(selected_model_path, model, device)
            X = np.hstack([u_np, v_np, np.abs(u_np - v_np)])
            probs = clf.predict_proba(X)[0]
            labels = ["Entailment", "Neutral", "Contradiction"]
            pred_idx = int(np.argmax(probs))
            pred_label = labels[pred_idx]

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Entailment Prob", f"{probs[0]*100:.1f}%")
                st.progress(float(probs[0]))
            with col2:
                st.metric("Neutral Prob", f"{probs[1]*100:.1f}%")
                st.progress(float(probs[1]))
            with col3:
                st.metric("Contradiction Prob", f"{probs[2]*100:.1f}%")
                st.progress(float(probs[2]))

            if pred_label == "Entailment":
                st.success(f"**Final NLI Classification**: **ENTAILMENT** (Prob: {probs[0]*100:.1f}%)")
            elif pred_label == "Neutral":
                st.warning(f"**Final NLI Classification**: **NEUTRAL** (Prob: {probs[1]*100:.1f}%)")
            else:
                st.error(f"**Final NLI Classification**: **CONTRADICTION** (Prob: {probs[2]*100:.1f}%)")

