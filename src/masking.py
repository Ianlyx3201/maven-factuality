"""Compare factuality evidence masking with nearby unannotated control spans.

Example (initial pipeline check):
    python src/masking.py --checkpoint models/roberta-smoke --split dev --limit 40
"""

import argparse
import csv
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def marked_text(words, trigger_start, trigger_end, masked, mask_token):
    """Reproduce training-time event markers, replacing selected source words."""
    updated = [mask_token if i in masked else word for i, word in enumerate(words)]
    updated.insert(trigger_end, "</EVT>")
    updated.insert(trigger_start, "<EVT>")
    return " ".join(updated)


def controls_for(example, count):
    """Select same-length spans with closest event-distance, without overlap."""
    words = example["tokens"]
    start, end = example["trigger_start"], example["trigger_end"]
    evidence = sorted(example["evidence_indices"])
    size = len(evidence)
    if not size or evidence != list(range(evidence[0], evidence[0] + size)):
        return []  # This initial version handles contiguous evidence spans only.
    if not (0 <= start < end <= len(words)):
        return []
    if evidence[0] < 0 or evidence[-1] >= len(words):
        return []
    if any(start <= i < end for i in evidence):
        return []

    trigger_mid = (start + end - 1) / 2
    evidence_dist = abs((evidence[0] + evidence[-1]) / 2 - trigger_mid)
    candidates = []
    for i in range(len(words) - size + 1):
        positions = list(range(i, i + size))
        if any(start <= j < end or j in evidence for j in positions):
            continue
        # Avoid pure punctuation spans, but not all possible linguistic confounds.
        if not all(any(character.isalnum() for character in words[j]) for j in positions):
            continue
        control_dist = abs((positions[0] + positions[-1]) / 2 - trigger_mid)
        candidates.append((abs(control_dist - evidence_dist), i, positions))

    candidates.sort(key=lambda item: (item[0], item[1]))
    return [indices for _, _, indices in candidates[:count]]


def prepare_examples(path, tokenizer, max_length, n_controls, limit):
    prepared = []
    counts = {"total": 0, "no_evidence_or_control": 0, "too_long": 0}
    mask_token = tokenizer.mask_token
    if not mask_token:
        raise ValueError("The tokenizer has no mask token.")

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            counts["total"] += 1
            row = json.loads(line)
            controls = controls_for(row, n_controls)
            if not controls:
                counts["no_evidence_or_control"] += 1
                continue
            words = row["tokens"]
            start, end = row["trigger_start"], row["trigger_end"]
            evidence = set(row["evidence_indices"])
            texts = [marked_text(words, start, end, set(), mask_token)]
            texts.append(marked_text(words, start, end, evidence, mask_token))
            texts.extend(marked_text(words, start, end, set(c), mask_token) for c in controls)
            if texts[0] != row["text"]:
                raise ValueError("Input disagrees with preprocess.py output; check event markers.")
            # Do not silently truncate away the marked target or any evidence.
            if any(len(tokenizer.encode(s, add_special_tokens=True)) > max_length for s in texts):
                counts["too_long"] += 1
                continue
            row["comparison_texts"] = texts
            row["control_indices"] = controls
            prepared.append(row)
            if limit > 0 and len(prepared) >= limit:
                break
    return prepared, counts


@torch.inference_mode()
def predict_probs(texts, tokenizer, model, device, batch_size, max_length):
    output = []
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(
            texts[start:start + batch_size],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batch = {key: value.to(device) for key, value in batch.items()}
        logits = model(**batch).logits
        output.extend(torch.softmax(logits, dim=-1).cpu().tolist())
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("models/roberta-smoke"))
    parser.add_argument("--split", choices=["dev", "valid"], default="dev")
    parser.add_argument("--limit", type=int, default=40, help="0 = all eligible examples")
    parser.add_argument("--n-controls", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--output", type=Path, default=Path("outputs/masking_pilot.csv"))
    args = parser.parse_args()
    if args.n_controls < 1:
        parser.error("--n-controls must be at least 1")

    data_path = Path("data/processed") / (args.split + ".jsonl")
    if not data_path.is_file():
        parser.error(f"Missing preprocessed data: {data_path}")
    if not (args.checkpoint / "config.json").is_file():
        parser.error(f"Missing model checkpoint: {args.checkpoint}")

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint)
    if not {"<EVT>", "</EVT>"}.issubset(set(tokenizer.get_added_vocab())):
        raise ValueError("Tokenizer is missing the event markers saved during training.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    print(f"Device: {device} | Checkpoint: {args.checkpoint}", flush=True)

    rows, counts = prepare_examples(data_path, tokenizer, args.max_length, args.n_controls, args.limit)
    if not rows:
        raise ValueError(f"No eligible examples. Filtering counts: {counts}")
    texts = [s for row in rows for s in row["comparison_texts"]]
    probs = predict_probs(texts, tokenizer, model, device, args.batch_size, args.max_length)

    results = []
    cursor = 0
    label2id = model.config.label2id
    for row in rows:
        n = len(row["comparison_texts"])
        per_input = probs[cursor:cursor + n]
        cursor += n
        gold = row["label"]
        if gold not in label2id:
            raise ValueError(f"Unknown model label: {gold}")
        label_index = label2id[gold]
        original = per_input[0][label_index]
        evidence_masked = per_input[1][label_index]
        control_mean = sum(p[label_index] for p in per_input[2:]) / (n - 2)
        delta_e = original - evidence_masked
        delta_c = original - control_mean
        results.append({
            "document_id": row["document_id"],
            "mention_id": row["mention_id"],
            "label": gold,
            "originally_correct": int(max(range(len(per_input[0])), key=per_input[0].__getitem__) == label_index),
            "evidence_words": " ".join(row["tokens"][i] for i in row["evidence_indices"]),
            "n_controls": n - 2,
            "p_original": original,
            "p_evidence_masked": evidence_masked,
            "p_control_masked_mean": control_mean,
            "delta_evidence": delta_e,
            "delta_control": delta_c,
            "difference": delta_e - delta_c,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    n = len(results)
    n_correct = sum(row["originally_correct"] for row in results)
    mean = lambda key: sum(row[key] for row in results) / n
    print(f"Eligible examples evaluated: {n}; originally correct: {n_correct}")
    print(f"Mean evidence probability drop: {mean('delta_evidence'):.4f}")
    print(f"Mean control probability drop:  {mean('delta_control'):.4f}")
    print(f"Mean paired difference:         {mean('difference'):.4f}")
    print(f"Filtering counts: {counts}")
    print(f"Saved per-example results to {args.output}")
    print("Note: initial control matching is approximate; no claim of semantic equivalence.")


if __name__ == "__main__":
    main()
