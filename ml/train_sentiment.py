# ml/train_sentiment.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments)
import torch
from datasets import Dataset

MODEL_NAME = "distilbert-base-uncased"
SAVE_DIR = "ml/sentiment_model"

label2id = {"negative": 0, "neutral": 1, "positive": 2}
id2label = {v: k for k, v in label2id.items()}

def load_data(path="data/ideas.csv"):
    df = pd.read_csv(path)
    df["label"] = df["sentiment"].map(label2id)
    return train_test_split(df, test_size=0.2, random_state=42)

def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    train_df, val_df = load_data()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=id2label,
        label2id=label2id,
    )

    train_ds = Dataset.from_pandas(train_df[["text", "label"]])
    val_ds   = Dataset.from_pandas(val_df[["text", "label"]])

    train_ds = train_ds.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    val_ds   = val_ds.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    train_ds = train_ds.remove_columns(["text"])
    val_ds   = val_ds.remove_columns(["text"])
    train_ds.set_format("torch")
    val_ds.set_format("torch")

    args = TrainingArguments(
        output_dir="ml/_sentiment_runs",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",
        learning_rate=3e-5,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=20,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"✅ Sentiment model saved to {SAVE_DIR}")

if __name__ == "__main__":
    main()
