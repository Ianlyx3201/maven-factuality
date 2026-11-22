"""Summarize paired masking effects with document-cluster bootstrap intervals.

Run from the repository root:
    python src/analyze_results.py --input outputs/masking_dev_all_1control.csv
"""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

LABELS = ("CT+", "CT-", "PS+", "PS-", "Uu")


def load_rows(path):
    required = {
        "document_id", "label", "originally_correct",
        "delta_evidence", "delta_control", "difference",
    }
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing CSV columns: {sorted(missing)}")
        rows = []
        for index, row in enumerate(reader, start=2):
            try:
                item = {
                    "document_id": row["document_id"],
                    "label": row["label"],
                    "originally_correct": int(row["originally_correct"]),
                    "delta_evidence": float(row["delta_evidence"]),
                    "delta_control": float(row["delta_control"]),
                    "difference": float(row["difference"]),
                }
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid values on CSV line {index}") from exc
            if item["label"] not in LABELS:
                raise ValueError(f"Unknown label on CSV line {index}")
            if item["originally_correct"] not in (0, 1):
                raise ValueError(f"Invalid correctness flag on line {index}")
            if abs(item["delta_evidence"] - item["delta_control"] - item["difference"]) > 1e-5:
                raise ValueError(f"Inconsistent paired difference on CSV line {index}")
            rows.append(item)
    if not rows:
        raise ValueError("The results CSV is empty")
    return rows


def percentile(sorted_values, fraction):
    index = fraction * (len(sorted_values) - 1)
    low = int(index)
    high = min(low + 1, len(sorted_values) - 1)
    weight = index - low
    return sorted_values[low] * (1 - weight) + sorted_values[high] * weight


def cluster_bootstrap_ci(rows, trials, seed):
    """Resample documents, retaining all their event mentions together."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["document_id"]].append(row["difference"])
    if len(groups) < 5 or trials == 0:
        return None

    clusters = [values for values in groups.values()]
    rng = random.Random(seed)
    estimates = []
    for _ in range(trials):
        total = 0.0
        n = 0
        for _ in range(len(clusters)):
            selected = clusters[rng.randrange(len(clusters))]
            total += sum(selected)
            n += len(selected)
        estimates.append(total / n)
    estimates.sort()
    return [percentile(estimates, 0.025), percentile(estimates, 0.975)]


def summarize(rows, trials, seed):
    if not rows:
        return None
    n = len(rows)
    return {
        "n_mentions": n,
        "n_documents": len({r["document_id"] for r in rows}),
        "originally_correct": sum(r["originally_correct"] for r in rows),
        "positive_d": sum(r["difference"] > 0 for r in rows),
        "mean_evidence_drop": sum(r["delta_evidence"] for r in rows) / n,
        "mean_control_drop": sum(r["delta_control"] for r in rows) / n,
        "mean_d": sum(r["difference"] for r in rows) / n,
        "document_bootstrap_ci95": cluster_bootstrap_ci(rows, trials, seed),
    }


def show(name, result):
    if result is None:
        print(f"{name}: no examples")
        return
    ci = result["document_bootstrap_ci95"]
    ci_text = "not estimated" if ci is None else f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"
    print(
        f"{name}: n={result['n_mentions']}, docs={result['n_documents']}, "
        f"correct={result['originally_correct']}, positive D={result['positive_d']}, "
        f"mean D={result['mean_d']:+.4f}, 95% cluster CI={ci_text}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("outputs/masking_dev_all_1control.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/masking_dev_summary.json"))
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.bootstrap < 0:
        parser.error("--bootstrap must be nonnegative")

    rows = load_rows(args.input)
    overall = summarize(rows, args.bootstrap, args.seed)
    correct = summarize([r for r in rows if r["originally_correct"]], args.bootstrap, args.seed + 1)
    by_label = {
        label: summarize([r for r in rows if r["label"] == label], args.bootstrap, args.seed + i + 2)
        for i, label in enumerate(LABELS)
    }
    print(f"Input: {args.input}")
    show("Overall", overall)
    print(f"  Evidence drop={overall['mean_evidence_drop']:+.4f}; control drop={overall['mean_control_drop']:+.4f}")
    show("Originally correct", correct)
    print("By gold label:")
    for label in LABELS:
        show("  " + label, by_label[label])

    result = {
        "input_csv": str(args.input),
        "bootstrap_trials": args.bootstrap,
        "resampling_unit": "document_id",
        "overall": overall,
        "originally_correct_subset": correct,
        "by_label": by_label,
        "limitations": (
            "Development data and evidence-annotated eligible subset only. "
            "Intervals reflect document resampling, not uncertainty from model training or control selection. "
            "Matched controls may contain relevant cues; masking may introduce distribution shift."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Saved summary to {args.output}")


if __name__ == "__main__":
    main()
