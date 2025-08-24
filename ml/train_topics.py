# ml/train_topics.py
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments)
from datasets import Dataset
from ml.labels import TOPIC_LABELS

MODEL_NAME = "distilbert-base-uncased"
SAVE_DIR = "ml/topic_model"

label2id = {label: i for i, label in enumerate(TOPIC_LABELS)}
id2label = {v: k for k, v in label2id.items()}

def encode_labels(topic_str):
    multi_hot = np.zeros(len(TOPIC_LABELS), dtype=int)
    if isinstance(topic_str, str):
        for t in topic_str.split("|"):
            t = t.strip()
            if t in label2id:
                multi_hot[label2id[t]] = 1
    return multi_hot.tolist()

def load_data(path="data/ideas_topics.csv"):
    df = pd.read_csv(path)
    df["labels"] = df["topics"].apply(encode_labels)
    return train_test_split(df, test_size=0.2, random_state=42)

def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    train_df, val_df = load_data()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(TOPIC_LABELS),
        problem_type="multi_label_classification",
        id2label=id2label,
        label2id=label2id,
    )

    train_ds = Dataset.from_pandas(train_df[["text", "labels"]])
    val_ds   = Dataset.from_pandas(val_df[["text", "labels"]])

    train_ds = train_ds.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    val_ds   = val_ds.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    def set_format(ds):
        ds = ds.remove_columns(["text"])
        ds.set_format(type="torch")
        return ds

    train_ds = set_format(train_ds)
    val_ds   = set_format(val_ds)

    args = TrainingArguments(
        output_dir="ml/_topic_runs",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",
        learning_rate=3e-5,
        num_train_epochs=4,
        weight_decay=0.01,
        logging_steps=20,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss"
    )

    def compute_loss(model, inputs, return_outputs=False):
        # Replace default loss with BCEWithLogits for multi-label
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.logits
        loss_fn = torch.nn.BCEWithLogitsLoss()
        loss = loss_fn(logits, labels.float())
        return (loss, outputs) if return_outputs else loss

    # Small wrapper to plug a custom loss into Trainer
    from transformers.trainer import Trainer as HFTrainer
    class MyTrainer(HFTrainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            return compute_loss(model, inputs, return_outputs)

    trainer = MyTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"✅ Topic model saved to {SAVE_DIR}")

if __name__ == "__main__":
    import torch  # make sure torch is available
    main()
