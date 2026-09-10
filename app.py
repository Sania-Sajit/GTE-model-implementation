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

            def encode(self, sentences, device=device, batch_size=32, normalize=True, **kwargs):
                all_embs = []
                with torch.no_grad():
                    for i in range(0, len(sentences), batch_size):
                        batch = sentences[i : i + batch_size]
                        encoded = self.tok(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
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


# ─────────────────────────────────────────────────────────────────────────────
# 3. Sidebar Configuration
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.title("⚡ GTE Dashboard Settings")
st.sidebar.markdown("---")

available_checkpoints = {}

if os.path.exists("gte_mini_model"):
    available_checkpoints["GTE-Mini (Stage 2 Model)"] = "gte_mini_model"
if os.path.exists("gte_mt_model"):
    available_checkpoints["GTE-MT (English-German Best Model)"] = "gte_mt_model"
if os.path.exists("gte_mt_hi_model"):
    available_checkpoints["GTE-MT (English-Hindi Best Model)"] = "gte_mt_hi_model"
if os.path.exists("checkpoints/mt_en_de/best_checkpoint"):
    available_checkpoints["EN-DE Best Checkpoint (Step 473)"] = "checkpoints/mt_en_de/best_checkpoint"
if os.path.exists("checkpoints/mt_en_hi/best_checkpoint"):
    available_checkpoints["EN-HI Best Checkpoint (Step 479)"] = "checkpoints/mt_en_hi/best_checkpoint"
if os.path.exists("checkpoints/stage1/best_checkpoint"):
    available_checkpoints["Stage 1 Pre-trained Checkpoint"] = "checkpoints/stage1/best_checkpoint"

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

tab_leaderboard, tab_training, tab_playground = st.tabs([
    "📊 Benchmark Leaderboard",
    "📈 Training Loss Monitor",
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
    st.subheader("🌐 Phase 4: Machine Translation Cross-Lingual Retrieval (Recall@k %)")

    mt_models = ["GTE-Mini (Base)", "GTE-MT (Fine-Tuned)"]
    r1_en2de = [2.20, 23.60]
    r5_en2de = [5.60, 45.60]
    r10_en2de = [7.60, 60.40]

    mt_de_json = "checkpoints/mt_results_en_de.json"
    if os.path.exists(mt_de_json):
        try:
            with open(mt_de_json) as f:
                mt_de_data = json.load(f)
            if "GTE-Mini (Base)" in mt_de_data and "En->De" in mt_de_data["GTE-Mini (Base)"]:
                r1_en2de[0] = round(float(mt_de_data["GTE-Mini (Base)"]["En->De"]["1"]), 2)
                r5_en2de[0] = round(float(mt_de_data["GTE-Mini (Base)"]["En->De"]["5"]), 2)
                r10_en2de[0] = round(float(mt_de_data["GTE-Mini (Base)"]["En->De"]["10"]), 2)
            ft_keys = [k for k in mt_de_data.keys() if "Fine-Tuned" in k or "GTE-MT" in k]
            if ft_keys:
                k = ft_keys[0]
                if "En->De" in mt_de_data[k]:
                    r1_en2de[1] = round(float(mt_de_data[k]["En->De"]["1"]), 2)
                    r5_en2de[1] = round(float(mt_de_data[k]["En->De"]["5"]), 2)
                    r10_en2de[1] = round(float(mt_de_data[k]["En->De"]["10"]), 2)
        except Exception:
            pass

    fig_mt = go.Figure(data=[
        go.Bar(name='Recall@1 (%)', x=mt_models, y=r1_en2de, marker_color='#43A047'),
        go.Bar(name='Recall@5 (%)', x=mt_models, y=r5_en2de, marker_color='#FB8C00'),
        go.Bar(name='Recall@10 (%)', x=mt_models, y=r10_en2de, marker_color='#E53935'),
    ])
    fig_mt.update_layout(title="English → German Sentence Retrieval Recall@k", barmode='group', yaxis=dict(range=[0, 100]))
    st.plotly_chart(fig_mt)



# =============================================================================
# TAB 2: Training Loss Monitor
# =============================================================================
with tab_training:
    st.subheader("📈 Multi-Stage Training Analytics")
    st.markdown("Inspect training loss progression across Stage 1 (pre-training), Stage 2 (fine-tuning), and Phase 4 (MT fine-tuning).")

    stage1_json = "checkpoints/stage1/last_checkpoint/training_state.json"
    stage2_json = "checkpoints/stage2/last_checkpoint/training_state.json"

    col_t1, col_t2 = st.columns(2)

    with col_t1:
        st.markdown("#### Stage 1 Pre-Training Loss Curve")
        if os.path.exists(stage1_json):
            with open(stage1_json) as f:
                state1 = json.load(f)
            steps1 = state1.get("steps", [])
            losses1 = state1.get("losses", [])
            if steps1 and losses1:
                fig1 = px.line(x=steps1, y=losses1, labels={'x': 'Step', 'y': 'Loss'}, title="Stage 1 ICL Loss Progression")
                fig1.update_traces(line_color="#1E88E5", line_width=2.5)
                st.plotly_chart(fig1)
            else:
                st.info("Stage 1 metrics logging in progress.")
        else:
            st.info("Run `python3 train.py` to generate Stage 1 training logs.")

    with col_t2:
        st.markdown("#### Stage 2 Fine-Tuning Loss Curve")
        if os.path.exists(stage2_json):
            with open(stage2_json) as f:
                state2 = json.load(f)
            steps2 = state2.get("steps", [])
            losses2 = state2.get("losses", [])
            if steps2 and losses2:
                fig2 = px.line(x=steps2, y=losses2, labels={'x': 'Step', 'y': 'Loss'}, title="Stage 2 Supervised Fine-Tuning Loss")
                fig2.update_traces(line_color="#7B1FA2", line_width=2.5)
                st.plotly_chart(fig2)
            else:
                st.info("Stage 2 metrics logging in progress.")
        else:
            st.info("Run Stage 2 training to generate Stage 2 loss logs.")


# =============================================================================
# TAB 3: Interactive Inference Playground
# =============================================================================
with tab_playground:
    st.subheader(f"🎮 Interactive Inference Playground (`{selected_model_name}`)")

    model, tokenizer, device, m_type = load_checkpoint_model(selected_model_path)

    tool_choice = st.radio("Select Playground Tool:", [
        "1. Semantic Textual Similarity (STS) & Heatmap",
        "2. English ↔ German Cross-Lingual MT Retrieval Search",
        "3. NLI Premise-Hypothesis Entailment Tester"
    ])

    st.markdown("---")

    # 1. STS Tool
    if "1. Semantic" in tool_choice:
        st.markdown("### 🔍 Sentence Pair Similarity & Matrix Heatmap")

        input_type = st.radio("Input Mode:", ["Pair Comparison", "Multi-Sentence Heatmap Matrix"])

        if input_type == "Pair Comparison":
            s1 = st.text_input("Sentence 1:", "The developer is writing Python code for an AI model.")
            s2 = st.text_input("Sentence 2:", "A programmer is coding a machine learning algorithm in Python.")

            if st.button("Compute Cosine Similarity"):
                embs = model.encode([s1, s2], device=device, normalize=True)
                sim = float(torch.dot(embs[0], embs[1]).item()) * 100.0

                st.metric(label="Cosine Similarity Score", value=f"{sim:.2f}%")
                st.progress(max(0.0, min(1.0, sim / 100.0)))

        else:
            sentences_text = st.text_area(
                "Enter sentences (one per line):",
                "The cat sat on the soft mat.\nA kitten is resting on the rug.\nStock market index rose sharply today.\nInvestors celebrated Wall Street gains."
            )
            sent_list = [line.strip() for line in sentences_text.strip().split("\n") if line.strip()]

            if st.button("Generate Similarity Matrix") and sent_list:
                embs = model.encode(sent_list, device=device, normalize=True)
                sim_matrix = (embs @ embs.T).cpu().numpy()

                fig_hm = px.imshow(
                    sim_matrix,
                    x=[f"S{i+1}" for i in range(len(sent_list))],
                    y=[f"S{i+1}" for i in range(len(sent_list))],
                    text_auto=".2f",
                    color_continuous_scale="Viridis",
                    title="Sentence Cosine Similarity Matrix"
                )
                st.plotly_chart(fig_hm)

                st.markdown("**Sentence Legend:**")
                for i, s in enumerate(sent_list):
                    st.write(f"**S{i+1}**: {s}")

    # 2. MT Tool
    elif "2. English" in tool_choice or "Cross-Lingual" in tool_choice:
        st.markdown("### 🌐 Cross-Lingual Search & Retrieval")

        col_mt1, col_mt2 = st.columns(2)
        with col_mt1:
            lang_pair_choice = st.selectbox("Language Pair:", ["English ↔ German", "English ↔ Hindi"])
        with col_mt2:
            direction_choice = st.radio("Retrieval Direction:", ["English → Target", "Target → English"], horizontal=True)

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

        premise = st.text_input("Premise:", "A man is playing guitar on a street corner.")
        hypothesis = st.text_input("Hypothesis:", "A person is making music outdoors.")

        if st.button("Classify Relationship"):
            u = model.encode([premise], device=device, normalize=True)
            v = model.encode([hypothesis], device=device, normalize=True)

            sim = float(torch.dot(u[0], v[0]).item())

            st.write(f"Cosine Similarity between Premise & Hypothesis: **{sim:.4f}**")
            if sim > 0.65:
                st.success("Prediction: **ENTAILMENT** (High semantic similarity)")
            elif sim > 0.35:
                st.warning("Prediction: **NEUTRAL** (Moderate topic overlap)")
            else:
                st.error("Prediction: **CONTRADICTION** (Low / conflicting similarity)")
