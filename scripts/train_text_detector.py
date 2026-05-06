#!/usr/bin/env python3
"""
train_text_detector.py - fixed: guarded renaming + CSV encoding fallback
"""
import os
import argparse
import random
import numpy as np
from typing import Optional
import torch
from datasets import load_dataset, Dataset, DatasetDict
# load_metric used to live in datasets; now use evaluate.load as a fallback
try:
    from datasets import load_metric
except Exception:
    try:
        from evaluate import load as load_metric
    except Exception:
        load_metric = None

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def read_input_dataset(path: str, text_col: str, label_col: str, filetype: Optional[str]):
    ext = filetype or os.path.splitext(path)[1].lstrip('.').lower()
    if ext in ("csv",):
        try:
            ds = load_dataset("csv", data_files=path, split="train")
        except Exception:
            # fallback: use pandas with latin1 encoding
            import pandas as _pd
            df = _pd.read_csv(path, encoding="latin1")
            ds = Dataset.from_pandas(df)
    elif ext in ("tsv","txt"):
        try:
            ds = load_dataset("csv", data_files=path, split="train", delimiter="\t")
        except Exception:
            import pandas as _pd
            df = _pd.read_csv(path, delimiter="\t", encoding="latin1")
            ds = Dataset.from_pandas(df)
    elif ext in ("jsonl","json"):
        ds = load_dataset("json", data_files=path, split="train")
    else:
        raise ValueError("Unsupported file extension: " + ext)

    # make sure requested columns exist
    if text_col not in ds.column_names or label_col not in ds.column_names:
        raise ValueError(f"Columns not found in dataset. Available: {ds.column_names}")

    # Guarded renaming: only rename if source != target, and avoid conflicts
    if text_col != "text":
        if "text" in ds.column_names:
            raise ValueError(
                f"Rename conflict: dataset already contains a 'text' column but text_col='{text_col}'. "
                f"Either rename/remove the existing 'text' column or set --text-col to 'text'. "
                f"Available columns: {ds.column_names}"
            )
        ds = ds.rename_column(text_col, "text")
    if label_col != "label":
        if "label" in ds.column_names:
            raise ValueError(
                f"Rename conflict: dataset already contains a 'label' column but label_col='{label_col}'. "
                f"Either rename/remove the existing 'label' column or set --label-col to 'label'. "
                f"Available columns: {ds.column_names}"
            )
        ds = ds.rename_column(label_col, "label")

    def to_int_label(example):
        example["label"] = int(example["label"])
        return example
    ds = ds.map(to_int_label)
    return ds

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    probs = torch.nn.functional.softmax(torch.tensor(pred.predictions), dim=1).numpy()[:,1]
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = float("nan")
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "roc_auc": auc}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to CSV/TSV/JSONL with text+label")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--model_name", default="roberta-base")
    parser.add_argument("--output_dir", default="models/text_detector")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading dataset from", args.input)
    ds_all = read_input_dataset(args.input, args.text_col, args.label_col, None)

    df = ds_all.to_pandas()
    train_df, temp_df = train_test_split(df, test_size=(args.test_size + args.val_size), stratify=df["label"], random_state=args.seed)
    val_size_rel = args.val_size / (args.val_size + args.test_size) if (args.val_size + args.test_size) > 0 else 0.0
    val_df, test_df = train_test_split(temp_df, test_size=(1.0 - val_size_rel), stratify=temp_df["label"], random_state=args.seed)

    dataset = DatasetDict({
        "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
        "validation": Dataset.from_pandas(val_df.reset_index(drop=True)),
        "test": Dataset.from_pandas(test_df.reset_index(drop=True)),
    })

    print("Sizes:", {k: len(dataset[k]) for k in dataset})

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, padding=False, max_length=args.max_length)

    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        do_train=True,
        do_eval=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size*2,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        save_total_limit=args.save_total_limit,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=args.fp16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting training on device:", args.device)
    trainer.train()
    print("Evaluating on test set...")
    metrics = trainer.predict(dataset["test"])
    print("Test metrics:", metrics.metrics)

    print("Saving model to", args.output_dir)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    with open(os.path.join(args.output_dir, "label_map.json"), "w") as fh:
        fh.write('{"0":"real","1":"synth"}')

    print("All done.")

if __name__ == "__main__":
    main()
