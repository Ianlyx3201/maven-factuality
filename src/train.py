"""Fine-tune RoBERTa on MAVEN-FACT event factuality.

Examples:
    python src/train.py --train-limit 64 --dev-limit 32 --max-steps 8
    python src/train.py --epochs 2 --batch-size 8 --sampling sqrt \
        --output-dir models/roberta-balanced

The official validation set is never used for training or model selection.
"""

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

import torch
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABELS = ["CT+", "CT-", "PS+", "PS-", "Uu"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
MODEL_NAME = "FacebookAI/roberta-base"


def load_examples(path, limit, seed=42):
    if not path.is_file():
        raise FileNotFoundError(f"Missing preprocessed data: {path}")
    with path.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    if limit <= 0 or len(rows) <= limit:
        return rows

    # Stratification is only for smoke tests with explicit limits.
    rng = random.Random(seed)
    groups = {label: [] for label in LABELS}
    for row in rows:
        groups[row["label"]].append(row)
    for group in groups.values():
        rng.shuffle(group)

    selected = []
    quota = limit // len(LABELS)
    for label in LABELS:
        selected.extend(groups[label][:quota])

    chosen_ids = {id(row) for row in selected}
    remainder = [row for row in rows if id(row) not in chosen_ids]
    rng.shuffle(remainder)
    selected.extend(remainder[: limit - len(selected)])
    rng.shuffle(selected)
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-limit", type=int, default=0, help="0 = complete train split")
    parser.add_argument("--dev-limit", type=int, default=0, help="0 = complete dev split")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--max-steps", type=int, default=0, help="0 = all steps")
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--sampling", choices=["natural", "sqrt"], default="sqrt")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("models/roberta-balanced"))
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.eval_batch_size < 1:
        parser.error("epochs and batch sizes must be positive")
    if args.max_length < 16:
        parser.error("max-length must be at least 16")

    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    print(f"Device: {device} | mixed precision: {use_amp}", flush=True)

    train = load_examples(Path("data/processed/train.jsonl"), args.train_limit)
    dev = load_examples(Path("data/processed/dev.jsonl"), args.dev_limit)
    if not train or not dev:
        raise ValueError("Training and development splits must not be empty")

    train_counts = Counter(row["label"] for row in train)
    dev_counts = Counter(row["label"] for row in dev)
    print(f"Train: {len(train):,} {dict(train_counts)}", flush=True)
    print(f"Dev:   {len(dev):,} {dict(dev_counts)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<EVT>", "</EVT>"]})
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label={i: label for i, label in enumerate(LABELS)},
        label2id=LABEL_TO_ID,
    )
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    model.to(device)

    # The weighting affects TRAIN sampling only; dev retains its real distribution.
    # sqrt inverse frequency is less aggressive than fully balanced sampling.
    sampler = None
    if args.sampling == "sqrt":
        weights = [1.0 / math.sqrt(train_counts[row["label"]]) for row in train]
        sampler = WeightedRandomSampler(
            weights,
            num_samples=len(train),
            replacement=True,
            generator=torch.Generator().manual_seed(42),
        )

    def collate(rows):
        batch = tokenizer(
            [row["text"] for row in rows],
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        batch["labels"] = torch.tensor(
            [LABEL_TO_ID[row["label"]] for row in rows], dtype=torch.long
        )
        return batch

    train_loader = DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=collate,
        num_workers=0,
    )
    dev_loader = DataLoader(
        dev,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_score = float("-inf")
    all_metrics = []
    global_step = 0
    stop = False

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        epoch_steps = 0
        print(f"Starting epoch {epoch}/{args.epochs}", flush=True)

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            epoch_steps += 1
            running_loss += loss.detach().item()
            if global_step == 1 or global_step % args.log_every == 0:
                print(f"epoch={epoch} step={global_step} loss={loss.item():.4f}", flush=True)
            if args.max_steps > 0 and global_step >= args.max_steps:
                stop = True
                break

        model.eval()
        gold, predicted = [], []
        with torch.inference_mode():
            for batch in dev_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    logits = model(**batch).logits
                predicted.extend(logits.argmax(dim=-1).cpu().tolist())
                gold.extend(batch["labels"].cpu().tolist())

        score = f1_score(
            gold, predicted, labels=list(range(len(LABELS))),
            average="macro", zero_division=0,
        )
        metrics = {
            "epoch": epoch,
            "global_steps": global_step,
            "train_loss": running_loss / max(1, epoch_steps),
            "dev_macro_f1": score,
            "dev_prediction_counts": dict(Counter(LABELS[i] for i in predicted)),
            "dev_gold_counts": dict(dev_counts),
        }
        all_metrics.append(metrics)
        print(f"\nEpoch {epoch} | Dev macro-F1: {score:.4f}", flush=True)
        print(classification_report(
            gold, predicted, labels=list(range(len(LABELS))),
            target_names=LABELS, zero_division=0,
        ), flush=True)

        if score > best_score:
            best_score = score
            args.output_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            print(f"Saved best checkpoint to {args.output_dir}", flush=True)
        model.train()
        if stop:
            break

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "model": MODEL_NAME,
        "sampling": args.sampling,
        "training_examples": len(train),
        "development_examples": len(dev),
        "best_dev_macro_f1": best_score,
        "training_history": all_metrics,
        "note": "Official validation split was not used for model selection.",
    }
    with (args.output_dir / "training_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Best dev macro-F1: {best_score:.4f}; checkpoint: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
