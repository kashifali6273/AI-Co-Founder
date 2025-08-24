from transformers import pipeline

# Load Hugging Face sentiment model
sentiment_analyzer = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    """
    Returns sentiment label (POSITIVE/NEGATIVE) for a given idea text.
    """
    result = sentiment_analyzer(text)[0]
    return result["label"]
