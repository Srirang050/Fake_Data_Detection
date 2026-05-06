# scripts/train_text_detector_hf.py
"""
Train using HuggingFace Trainer.
Usage:
python scripts/train_text_detector_hf.py --train data/splits/train.csv --validation data/splits/validation.csv --test data/splits/test.csv --output_dir models/text_detector_clean
"""
import argparse, os, random, numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def compute_metrics(pred):
    import torch as _torch
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    probs = _torch.nn.functional.softmax(_torch.tensor(pred.predictions), dim=1).numpy()[:,1]
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = float("nan")
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "roc_auc": auc}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--validation", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--model_name", default="roberta-base")
    p.add_argument("--output_dir", default="models/text_detector_clean")
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--fp16", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # load CSVs as HF Dataset
    ds = DatasetDict({
        "train": Dataset.from_pandas(__import__('pandas').read_csv(args.train)),
        "validation": Dataset.from_pandas(__import__('pandas').read_csv(args.validation)),
        "test": Dataset.from_pandas(__import__('pandas').read_csv(args.test))
    })

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, padding=False, max_length=args.max_length)
    ds = ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size*2,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=args.fp16,
        save_total_limit=2,
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    print("Evaluating on test set...")
    print(trainer.predict(ds["test"]).metrics)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "label_map.json"), "w") as fh:
        fh.write('{"0":"real","1":"synth"}')

if __name__ == "__main__":
    main()
