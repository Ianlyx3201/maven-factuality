
"""Prepare event-level MAVEN-FACT examples for RoBERTa."""

import argparse
import hashlib
import json
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

LABELS = {"CT+", "CT-", "PS+", "PS-", "Uu"}


def get_evidence_indices(mention, words):
    offsets = mention.get("evidence_offset") or []
    evidence_words = mention.get("evidence_word") or []
    sid = mention["sent_id"]
    start, end = mention["offset"]

    if not offsets or len(offsets) != len(evidence_words):
        return []

    indices = []

    for position, word in zip(offsets, evidence_words):
        if not isinstance(position, list) or len(position) != 2:
            return []

        evidence_sid, i = position

        if evidence_sid != sid or not isinstance(i, int):
            return []

        if not 0 <= i < len(words):
            return []

        if start <= i < end:
            return []

        if words[i].casefold() != str(word).casefold():
            return []

        indices.append(i)

    return sorted(set(indices))


def extract_examples(document):
    sentences = document.get("tokens", [])

    for event in document.get("events", []):
        for mention in event.get("mention", []):
            label = mention.get("factuality")
            sid = mention.get("sent_id")
            offset = mention.get("offset")

            if label not in LABELS or not isinstance(sid, int):
                continue

            if not 0 <= sid < len(sentences):
                continue

            words = sentences[sid]

            if not isinstance(words, list):
                continue

            if not isinstance(offset, list) or len(offset) != 2:
                continue

            start, end = offset

            if not all(isinstance(x, int) for x in (start, end)):
                continue

            if not 0 <= start < end <= len(words):
                continue

            marked = (
                words[:start]
                + ["<EVT>"]
                + words[start:end]
                + ["</EVT>"]
                + words[end:]
            )

            yield {
                "document_id": document["id"],
                "mention_id": mention.get("id"),
                "sentence_id": sid,
                "tokens": words,
                "trigger_start": start,
                "trigger_end": end,
                "text": " ".join(marked),
                "label": label,
                "evidence_indices": get_evidence_indices(mention, words),
            }


def is_dev_document(document_id):
    digest = hashlib.sha256(
        str(document_id).encode("utf-8")
    ).digest()

    return int.from_bytes(digest[:4], "big") % 10 == 0


def process_file(path, writers, training):
    document_counts = Counter()
    label_counts = {name: Counter() for name in writers}

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            document = json.loads(line)

            if training:
                split = (
                    "dev"
                    if is_dev_document(document["id"])
                    else "train"
                )
            else:
                split = "valid"

            document_counts[split] += 1

            for example in extract_examples(document):
                writers[split].write(
                    json.dumps(example, ensure_ascii=False) + "\n"
                )
                label_counts[split][example["label"]] += 1

    return document_counts, label_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
    )
    args = parser.parse_args()

    train_path = args.data_dir / "train.jsonl"
    valid_path = args.data_dir / "valid.jsonl"

    for path in (train_path, valid_path):
        if not path.is_file():
            parser.error(f"Missing dataset file: {path}")

    output_dir = args.data_dir / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    with ExitStack() as stack:
        writers = {
            name: stack.enter_context(
                (output_dir / f"{name}.jsonl").open(
                    "w", encoding="utf-8"
                )
            )
            for name in ("train", "dev", "valid")
        }

        train_docs, train_labels = process_file(
            train_path, writers, training=True
        )
        valid_docs, valid_labels = process_file(
            valid_path, writers, training=False
        )

    for split in ("train", "dev", "valid"):
        docs = train_docs if split != "valid" else valid_docs
        labels = train_labels if split != "valid" else valid_labels

        print(
            f"{split}: {docs[split]} documents, "
            f"{sum(labels[split].values())} mentions"
        )
        print(f"  Labels: {dict(labels[split])}")

    print(f"\nSaved processed data to: {output_dir}")


if __name__ == "__main__":
    main()
