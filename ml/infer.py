# ml/infer.py
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from ml.labels import TOPIC_LABELS

# Paths where the training scripts saved the models
SENTIMENT_DIR = "ml/sentiment_model"
TOPIC_DIR     = "ml/topic_model"

_device = "cuda" if torch.cuda.is_available() else "cpu"

# Lazy singletons
_tokenizer_sent = None
_model_sent = None
_tokenizer_topic = None
_model_topic = None

def _load_sentiment():
    global _tokenizer_sent, _model_sent
    if _model_sent is None:
        _tokenizer_sent = AutoTokenizer.from_pretrained(SENTIMENT_DIR)
        _model_sent = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_DIR).to(_device).eval()

def _load_topics():
    global _tokenizer_topic, _model_topic
    if _model_topic is None:
        _tokenizer_topic = AutoTokenizer.from_pretrained(TOPIC_DIR)
        _model_topic = AutoModelForSequenceClassification.from_pretrained(TOPIC_DIR).to(_device).eval()

def predict_sentiment(text: str) -> str:
    try:
        _load_sentiment()
        enc = _tokenizer_sent(text, return_tensors="pt", truncation=True, padding=True, max_length=256).to(_device)
        with torch.no_grad():
            logits = _model_sent(**enc).logits
        pred = int(torch.argmax(logits, dim=-1).cpu().numpy()[0])
        mapping = {0: "negative", 1: "neutral", 2: "positive"}
        return mapping.get(pred, "neutral")
    except Exception:
        return "neutral"

def predict_topics(text: str, threshold: float = 0.5) -> list[str]:
    try:
        _load_topics()
        enc = _tokenizer_topic(text, return_tensors="pt", truncation=True, padding=True, max_length=256).to(_device)
        with torch.no_grad():
            logits = _model_topic(**enc).logits
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        on = np.where(probs >= threshold)[0].tolist()
        # fallback: if none cross threshold, pick top-1
        if not on:
            on = [int(np.argmax(probs))]
        return [TOPIC_LABELS[i] for i in on]
    except Exception:
        return []
