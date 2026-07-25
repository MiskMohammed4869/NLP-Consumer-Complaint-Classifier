from __future__ import annotations

import json
import os
import re
import string
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv("MODEL_DIR", APP_DIR / "final_distilbert"))
MAX_LENGTH = 128

DEFAULT_LABELS = {
    0: "credit_card",
    1: "credit_reporting",
    2: "debt_collection",
    3: "mortgages_and_loans",
    4: "retail_banking",
}

LABEL_DISPLAY_NAMES = {
    "credit_card": "Credit Card",
    "credit_reporting": "Credit Reporting",
    "debt_collection": "Debt Collection",
    "mortgages_and_loans": "Mortgages & Loans",
    "retail_banking": "Retail Banking",
}

# Colors used for the progress bars in the "Top Predictions" list, in order.
BAR_COLORS = ["#22c55e", "#38bdf8", "#a855f7", "#f59e0b", "#ef4444"]

STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "ain", "all", "am", "an",
    "and", "any", "are", "aren", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can", "couldn", "d", "did",
    "didn", "do", "does", "doesn", "doing", "don", "down", "during", "each", "few",
    "for", "from", "further", "had", "hadn", "has", "hasn", "have", "haven",
    "having", "he", "her", "here", "hers", "herself", "him", "himself", "his",
    "how", "i", "if", "in", "into", "is", "isn", "it", "its", "itself", "just",
    "ll", "m", "ma", "me", "mightn", "more", "most", "mustn", "my", "myself",
    "needn", "no", "nor", "not", "now", "o", "of", "off", "on", "once", "only",
    "or", "other", "our", "ours", "ourselves", "out", "over", "own", "re", "s",
    "same", "shan", "she", "should", "shouldn", "so", "some", "such", "t", "than",
    "that", "the", "their", "theirs", "them", "themselves", "then", "there",
    "these", "they", "this", "those", "through", "to", "too", "under", "until",
    "up", "ve", "very", "was", "wasn", "we", "were", "weren", "what", "when",
    "where", "which", "while", "who", "whom", "why", "will", "with", "won",
    "wouldn", "y", "you", "your", "yours", "yourself", "yourselves",
}

st.set_page_config(
    page_title="Consumer Complaint Classifier",
    page_icon=":material/analytics:",
    layout="centered",
)


# --------------------------------------------------------------------------
# Styling: dark glassy card with a blue/purple glow, matching the mock-up.
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at 15% 0%, rgba(56, 90, 220, 0.25), transparent 45%),
                    radial-gradient(circle at 85% 100%, rgba(147, 51, 234, 0.25), transparent 45%),
                    #060714;
    }

    .block-container {
        max-width: 760px;
        padding-top: 2.5rem;
    }

    #MainMenu, footer, header {visibility: hidden;}

    .app-card {
        background: linear-gradient(180deg, rgba(17, 19, 40, 0.95), rgba(10, 11, 26, 0.95));
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 20px;
        padding: 36px 40px 40px 40px;
        box-shadow: 0 0 40px rgba(88, 60, 220, 0.15);
    }

    .app-title {
        color: #f5f6ff;
        font-size: 2.4rem;
        font-weight: 800;
        margin: 4px 0 6px 0;
    }
    .app-subtitle {
        color: #b0b4c4;
        font-size: 1.25rem;
        margin-bottom: 30px;
    }

    .field-label {
        color: #d1d5db;
        font-size: 1.15rem;
        font-weight: 500;
        margin-bottom: 10px;
    }

    .stTextArea textarea {
        background-color: #0d1024 !important;
        border: 1px solid rgba(99, 102, 241, 0.25) !important;
        border-radius: 12px !important;
        color: #e5e7eb !important;
        font-size: 1.15rem !important;
    }

    .stSelectbox > div > div {
        background-color: #0d1024 !important;
        border: 1px solid rgba(99, 102, 241, 0.25) !important;
        border-radius: 12px !important;
        color: #e5e7eb !important;
        font-size: 1.15rem !important;
    }

    .stButton > button, .stFormSubmitButton > button {
        width: 100%;
        background: linear-gradient(90deg, #4f46e5, #9333ea);
        color: white;
        font-weight: 700;
        font-size: 1.3rem;
        border: none;
        border-radius: 12px;
        padding: 1rem 0;
        box-shadow: 0 8px 24px rgba(99, 102, 241, 0.4);
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover {
        filter: brightness(1.08);
        color: white;
    }
    .stButton > button p, .stFormSubmitButton > button p {
        font-size: 1.3rem !important;
    }
    .stButton > button svg, .stFormSubmitButton > button svg {
        width: 1.5rem !important;
        height: 1.5rem !important;
    }

    .prediction-card {
        background: rgba(16, 60, 40, 0.35);
        border: 1px solid rgba(34, 197, 94, 0.35);
        border-radius: 16px;
        padding: 26px 28px;
        margin-top: 28px;
    }
    .prediction-label {
        color: #4ade80;
        font-size: 1.15rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 8px;
    }
    .prediction-value {
        color: #f5f6ff;
        font-size: 2.1rem;
        font-weight: 800;
    }
    .prediction-confidence {
        color: #b0b4c4;
        font-size: 1.15rem;
        margin-top: 6px;
    }

    .top-predictions-title {
        color: #f5f6ff;
        font-weight: 700;
        font-size: 1.4rem;
        margin: 30px 0 16px 0;
    }

    .bar-row {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 16px;
    }
    .bar-name {
        width: 190px;
        color: #d1d5db;
        font-size: 1.1rem;
        flex-shrink: 0;
    }
    .bar-track {
        flex-grow: 1;
        background: rgba(255, 255, 255, 0.06);
        border-radius: 8px;
        height: 12px;
        overflow: hidden;
    }
    .bar-fill {
        height: 100%;
        border-radius: 8px;
    }
    .bar-pct {
        width: 70px;
        text-align: right;
        color: #e5e7eb;
        font-size: 1.1rem;
        flex-shrink: 0;
    }

    .empty-state {
        color: #b0b4c4;
        font-size: 1.15rem;
        margin-top: 28px;
    }

    section[data-testid="stSidebar"] {
        background-color: #05060f;
        border-right: 1px solid rgba(99, 102, 241, 0.2);
    }
    section[data-testid="stSidebar"] * {
        font-size: 1.05rem;
        color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def clean_complaint(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"@\S+", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\.pic\S+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = "".join(char for char in text if char not in string.punctuation)
    words = [
        word
        for word in text.split()
        if word not in STOP_WORDS and len(word) > 2
    ]
    return re.sub(r"\s+", " ", " ".join(words)).strip()


def display_name(label: str) -> str:
    return LABEL_DISPLAY_NAMES.get(label, label.replace("_", " ").title())


@st.cache_data
def load_labels(model_dir: str) -> dict[int, str]:
    label_path = Path(model_dir) / "label_mapping.json"
    if not label_path.exists():
        label_path = APP_DIR / "label_mapping.json"

    if label_path.exists():
        with label_path.open("r", encoding="utf-8") as file:
            loaded = json.load(file)
        return {int(key): value for key, value in loaded.items()}

    return DEFAULT_LABELS


@st.cache_resource(show_spinner="Loading DistilBERT model...")
def load_model(model_dir: str):
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Missing ML dependencies. Install them with: pip install -r requirements.txt"
        ) from exc

    model_path = Path(model_dir)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model folder was not found at {model_path}. "
            "Place final_distilbert beside streamlit_app.py or set MODEL_DIR."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        local_files_only=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device, torch


def predict(text: str, labels: dict[int, str]) -> tuple[str, pd.DataFrame]:
    tokenizer, model, device, torch = load_model(str(MODEL_DIR))
    cleaned_text = clean_complaint(text)
    model_input = cleaned_text or str(text)

    encoded = tokenizer(
        model_input,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        probabilities = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

    rows = [
        {
            "category": labels.get(index, f"LABEL_{index}"),
            "confidence": float(probability) * 100,
        }
        for index, probability in enumerate(probabilities)
    ]
    results = pd.DataFrame(rows).sort_values("confidence", ascending=False).reset_index(drop=True)
    return cleaned_text, results


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------
st.markdown('<div class="app-card">', unsafe_allow_html=True)

st.markdown(
    """
    <div class="app-title">Consumer Complaint Classifier</div>
    <div class="app-subtitle">Enter a complaint and get instant classification</div>
    """,
    unsafe_allow_html=True,
)

labels = load_labels(str(MODEL_DIR))

with st.form("complaint-form", border=False):
    st.markdown('<div class="field-label">Enter your complaint</div>', unsafe_allow_html=True)
    complaint_text = st.text_area(
        "Complaint narrative",
        value="",
        height=140,
        placeholder="My credit card was charged twice for the same purchase and I have been trying to get a refund for a week.",
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button(
        "Predict",
        type="primary",
        icon=":material/send:",
    )

if submitted:
    if not complaint_text.strip():
        st.warning("Enter a complaint narrative before running prediction.", icon=":material/warning:")
        st.stop()

    try:
        cleaned_text, results = predict(complaint_text, labels)
    except Exception as exc:
        st.error(str(exc), icon=":material/error:")
        st.stop()

    top_result = results.iloc[0]

    st.markdown(
        f"""
        <div class="prediction-card">
            <div class="prediction-label">✅ Prediction</div>
            <div class="prediction-value">{display_name(str(top_result['category']))}</div>
            <div class="prediction-confidence">Confidence: {top_result['confidence']:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="top-predictions-title">Top Predictions</div>', unsafe_allow_html=True)

    bars_html = ""
    for i, row in results.iterrows():
        color = BAR_COLORS[i % len(BAR_COLORS)]
        bars_html += f"""
        <div class="bar-row">
            <div class="bar-name">{display_name(str(row['category']))}</div>
            <div class="bar-track">
                <div class="bar-fill" style="width:{row['confidence']:.1f}%; background:{color};"></div>
            </div>
            <div class="bar-pct">{row['confidence']:.1f}%</div>
        </div>
        """
    st.markdown(bars_html, unsafe_allow_html=True)

    with st.expander("Preprocessed text", icon=":material/visibility:"):
        st.write(cleaned_text or "No tokens remained after preprocessing.")
else:
    st.markdown(
        '<div class="empty-state">Paste a complaint narrative or choose an example, then click Predict.</div>',
        unsafe_allow_html=True,
    )

st.markdown("</div>", unsafe_allow_html=True)

with st.sidebar:
    st.subheader("Model", anchor=False)
    st.badge("DistilBERT", icon=":material/model_training:", color="blue")
    st.caption("Best validation result from the notebook.")
    st.metric("Accuracy", "89.52%")
    st.metric("F1-score", "89.04%")