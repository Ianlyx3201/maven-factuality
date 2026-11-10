
"""Inspect MAVEN-FACT data and supporting evidence annotations."""

import argparse
import json
from collections import Counter
from pathlib import Path


def evidence_is_usable(mention, tokens):
    """Check evidence alignment and exclude target-event overlap."""
    offsets = mention.get("evidence_offset") or []
    words = mention.get("evidence_word") or []
    sid = mention.get("sent_id")
    trigger = mention.get("offset") or []

    if not offsets or len(offsets) != len(words):
        return False
    if not isinstance(sid, int) or not (0 <= sid < len(tokens)):
        return False
    if not (isinstance(trigger, list) and len(trigger) == 2):
        return False

    for pos, word in zip(offsets, words):
        if not isinstance(pos, list) or len(pos) != 2:
            return False

        sent_idx, token_idx = pos

        if not (isinstance(sent_idx, int) and isinstance(token_idx, int)):
            return False
        if sent_idx != sid or not (0 <= token_idx < len(tokens[sid])):
            return False
        if trigger[0] <= token_idx < trigger[1]:
            return False
        if tokens[sid][token_idx].casefold() != str(word).casefold():
            return False

    return True


def inspect(path):
    documents = 0
    mentions = 0
    evidence_count = 0
    usable_count = 0
    labels = Counter()
    usable_labels = Counter()

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)
            documents += 1
            tokens = record["tokens"]

            for event in record.get("events", []):
                for mention in event.get("mention", []):
                    mentions += 1
                    label = mention.get("factuality", "MISSING")
                    labels[label] += 1

                    if mention.get("evidence_offset"):
                        evidence_count += 1

                        if evidence_is_usable(mention, tokens):
                            usable_count += 1
                            usable_labels[label] += 1

    print(f"\n{path.name}")
    print(f"Documents: {documents:,}")
    print(f"Event mentions: {mentions:,}")

    print("\nFactuality labels:")
    for label, count in labels.most_common():
        print(f"  {label}: {count:,} ({count / mentions:.1%})")

    print(f"\nMentions with evidence: {evidence_count:,}")
    print(f"Usable evidence (initial filter): {usable_count:,}")
    print(f"Usable by label: {dict(sorted(usable_labels.items()))}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
    )
    args = parser.parse_args()

    for name in ("train.jsonl", "valid.jsonl"):
        path = args.data_dir / name

        if not path.is_file():
            parser.error(f"Missing dataset file: {path}")

        inspect(path)


if __name__ == "__main__":
    main()
