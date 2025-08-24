# labeler.py

def assign_label(text: str) -> str:
    text = text.lower()
    if "ai" in text or "machine learning" in text:
        return "AI/ML"
    elif "finance" in text or "payment" in text:
        return "FinTech"
    elif "health" in text or "medical" in text:
        return "HealthTech"
    elif "education" in text or "learning" in text:
        return "EdTech"
    else:
        return "General"
