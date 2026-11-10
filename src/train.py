
"""Fine-tune RoBERTa for event factuality classification."""

import argparse
import json
import random
from pathlib import Path

import torch
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABELS = ["CT+", "CT-", "PS+", "PS-", "Uu"]
LABEL_TO_ID = {name: i for i, name in enumerate(LABELS)}
MODEL_NAME = "FacebookAI/roberta-base"


def load_examples(path, limit):
    with path.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    if limit <= 0 or len(rows) <= limit:
        return rows

    # Include minority classes in small pipeline tests.
    rng = random.Random(42)
    groups = {label: [] for label in LABELS}

    for row in rows:
        groups[row["label"]].append(row)

    for group in groups.values():
        rng.shuffle(group)

    selected = []
    for label in LABELS:
        selected.extend(groups[label][:limit // len(LABELS)])

    selected_ids = {id(row) for row in selected}
    remaining = [row for row in rows if id(row) not in selected_ids]
    rng.shuffle(remaining)
    selected.extend(remaining[:limit - len(selected)])
    rng.shuffle(selected)

    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--dev-limit", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/roberta-factuality"),
    )
    args = parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}", flush=True)

    train = load_examples(
        Path("data/processed/train.jsonl"), args.train_limit
    )
    dev = load_examples(
        Path("data/processed/dev.jsonl"), args.dev_limit
    )

    print(
        f"Training examples: {len(train)} | "
        f"Development examples: {len(dev)}",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, use_fast=True
    )
    tokenizer.add_special_tokens({
        "additional_special_tokens": ["<EVT>", "</EVT>"]
    })

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label={i: label for i, label in enumerate(LABELS)},
        label2id=LABEL_TO_ID,
    )
    model.resize_token_embeddings(
        len(tokenizer), mean_resizing=False
    )
    model.to(device)

    def collate(rows):
        batch = tokenizer(
            [row["text"] for row in rows],
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        batch["labels"] = torch.tensor(
            [LABEL_TO_ID[row["label"]] for row in rows],
            dtype=torch.long,
        )
        return batch

    train_loader = DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    dev_loader = DataLoader(
        dev,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2e-5
    )
    step = 0

    for epoch in range(args.epochs):
        model.train()

        for batch in train_loader:
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            loss = model(**batch).loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            step += 1

            if step == 1 or step % 2 == 0:
                print(
                    f"epoch={epoch + 1} "
                    f"step={step} loss={loss.item():.4f}",
                    flush=True,
                )

            if args.max_steps > 0 and step >= args.max_steps:
                break

        if args.max_steps > 0 and step >= args.max_steps:
            break

    model.eval()
    gold = []
    predicted = []

    with torch.no_grad():
        for batch in dev_loader:
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            outputs = model(**batch)
            predicted.extend(
                outputs.logits.argmax(dim=-1).cpu().tolist()
            )
            gold.extend(batch["labels"].cpu().tolist())

    macro_f1 = f1_score(
        gold,
        predicted,
        labels=list(range(len(LABELS))),
        average="macro",
        zero_division=0,
    )

    print(f"\nDev macro-F1: {macro_f1:.4f}")
    print(classification_report(
        gold,
        predicted,
        labels=list(range(len(LABELS))),
        target_names=LABELS,
        zero_division=0,
    ))

    args.output_dir.mkdir(
        parents=True, exist_ok=True
    )
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(
        f"Saved model and tokenizer to: {args.output_dir}"
    )


if __name__ == "__main__":
    main()
