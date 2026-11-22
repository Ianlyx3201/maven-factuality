# Evidence Sensitivity in Neural Event Factuality Prediction

LING 482 Term Project — Fall 2026

## Research question

Does a fine-tuned event factuality classifier respond more strongly to the removal of human-annotated supporting evidence than to the removal of comparable unannotated text?

**Hypothesis:** Masking annotated evidence will produce a larger decrease in the gold-label probability than masking a matched control span.

## Dataset

[MAVEN-FACT](https://aclanthology.org/2024.findings-emnlp.651/) (Li et al., 2024) supplies event factuality labels and evidence-word annotations for selected event mentions. Download `train.jsonl` and `valid.jsonl` from the [dataset files](https://drive.google.com/drive/folders/1GP8cdPlcUx5Om-lQse11LTDU-QFeAMom), then place them in `data/`:

```text
data/
  train.jsonl
  valid.jsonl
```

The dataset, processed files, model checkpoints, and experiment outputs are local and excluded from Git.

## Pipeline

| Script | Purpose |
|---|---|
| `src/inspect_data.py` | Inspect label frequencies and check evidence-word alignment. |
| `src/preprocess.py` | Mark target events with `<EVT>` tags; create a document-level train/dev split while retaining the official validation split. |
| `src/train.py` | Fine-tune RoBERTa-base for five-class classification; evaluate on dev and save the best checkpoint. |
| `src/masking.py` | Compare gold-label probabilities for original, evidence-masked, and control-masked input. |
| `src/analyze_results.py` | Summarize paired masking differences by label and estimate document-cluster bootstrap intervals. |

The classifier uses **sentence-level** context. The first masking implementation requires contiguous, sentence-aligned evidence that does not overlap the target event. Control spans match evidence in source-token length and approximately in distance from the target; this is an initial comparison, **not** a semantic or syntactic matching guarantee.

## Environment and commands

Developed with Python 3.11, CUDA-enabled PyTorch, Hugging Face Transformers, and scikit-learn on Windows with an NVIDIA RTX 4080 Laptop GPU. Run the commands below from the repository root after installing these packages.

Inspect and preprocess:

```bash
python src/inspect_data.py
python src/preprocess.py
```

Small training smoke test:

```bash
python src/train.py --train-limit 64 --dev-limit 32 --max-steps 8 --batch-size 4 --output-dir models/roberta-smoke-v2
```

Development-stage pilot training (1,200 optimizer steps; not full training):

```bash
python src/train.py --epochs 1 --batch-size 4 --max-steps 1200 --sampling sqrt --log-every 100 --output-dir models/roberta-pilot-v1
```

Run masking on all eligible **dev** examples with one control per example, then analyze the CSV:

```bash
python src/masking.py --checkpoint models/roberta-pilot-v1 --split dev --limit 0 --n-controls 1 --output outputs/masking_dev_all_1control.csv
python src/analyze_results.py --input outputs/masking_dev_all_1control.csv --output outputs/masking_dev_summary.json
```

The official `valid` split is reserved for later evaluation; current development-stage results are from the held-out portion of the training data (`dev`).

## Current development-stage results

Data inspection:

| Statistic | Original train | Official validation |
|---|---:|---:|
| Documents | 2,913 | 710 |
| Event mentions | 73,939 | 17,780 |
| Mentions with evidence annotations | 3,724 | 843 |
| Evidence passing initial alignment checks | 3,584 | 797 |

The document-level development split has 2,644 training documents (67,206 event mentions) and 269 dev documents (6,733 mentions).

The 1,200-step pilot model achieved **dev macro-F1 = 0.5134**. Its masking analysis over **348 eligible dev mentions** yielded a mean paired difference **D = +0.6148** (95% document-cluster bootstrap interval **[+0.5558, +0.6816]**). Here, `D` is the gold-label probability drop under evidence masking minus the drop under control masking. Among 276 originally correct examples, mean `D = +0.7591`.

These are **preliminary development results**, not final validation results. The eligible subset is limited and class-skewed; control spans have not been semantically matched or fully audited, and token masking may itself change the input distribution. The positive difference does not establish semantic understanding.

## Next steps

- Audit control quality and test sensitivity to control selection.
- Complete fuller model training and final held-out evaluation.
- If feasible, pilot manually verified edits that change event factuality.

## References

- Li et al. (2024). [MAVEN-FACT: A Large-scale Event Factuality Detection Dataset](https://aclanthology.org/2024.findings-emnlp.651/).
- DeYoung et al. (2020). [ERASER: A Benchmark to Evaluate Rationalized NLP Models](https://aclanthology.org/2020.acl-main.408/).
