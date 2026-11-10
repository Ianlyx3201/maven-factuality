
# Evidence Sensitivity in Neural Event Factuality Prediction

LING 482 Term Project, Fall 2026

## Research Question

Does a neural event factuality classifier rely more on human-annotated factuality evidence than on comparable unannotated text?

The main hypothesis is that masking annotated evidence will cause a larger decrease in gold-label probability than masking matched control spans.

## Dataset

This project uses [MAVEN-FACT](https://github.com/THU-KEG/MAVEN-FACT) (Li et al., 2024), which provides event factuality labels and supporting-word annotations.

Download `train.jsonl` and `valid.jsonl` from the [official dataset folder](https://drive.google.com/drive/folders/1GP8cdPlcUx5Om-lQse11LTDU-QFeAMom).

Place both files in a local `data/` directory:

```text
data/
  train.jsonl
  valid.jsonl
```

Dataset files and generated outputs are excluded from Git.

## Methods

1. Inspect factuality labels and supporting-evidence annotations.
2. Prepare event-marked inputs and split training documents into train/dev sets.
3. Fine-tune RoBERTa-base for five-class event factuality prediction.
4. Compare predictions on original inputs, evidence-masked inputs, and matched-control-masked inputs.
5. Measure gold-label probability changes, macro-F1, and per-class performance.
6. If feasible, test manually verified edits that change event factuality.

## Repository Structure

```text
src/
  inspect_data.py    Dataset statistics and evidence-alignment checks
  preprocess.py      Event-aware preprocessing and data splits
  train.py           RoBERTa fine-tuning and evaluation
  masking.py         Evidence and control masking (planned)
```

The first three scripts are implemented and tested. The masking experiment remains under development.

## Current Progress

The dataset inspection and preprocessing scripts have been tested on the original training and validation files.

| Statistic | Train | Validation |
|---|---:|---:|
| Documents | 2,913 | 710 |
| Event mentions | 73,939 | 17,780 |
| Mentions with evidence | 3,724 | 843 |
| Evidence passing initial alignment checks | 3,584 | 797 |

The preprocessing script also creates a document-level split:

| Split | Documents | Event mentions |
|---|---:|---:|
| Train | 2,644 | 67,206 |
| Dev | 269 | 6,733 |
| Validation | 710 | 17,780 |

The RoBERTa-base training and evaluation pipeline passed an initial GPU smoke test with 64 training examples, 32 development examples, and eight optimization steps. Training, evaluation, and checkpoint saving completed successfully.

This was a pipeline test, not a meaningful estimate of model performance. Full fine-tuning and masking experiments remain.

## Running the Code

Python 3.11 is used for development.

The inspection and preprocessing scripts require only the Python standard library. Model training additionally requires PyTorch, Transformers, and scikit-learn.

### 1. Inspect the dataset

```bash
python src/inspect_data.py
```

### 2. Prepare model inputs

```bash
python src/preprocess.py
```

This creates:

```text
data/processed/
  train.jsonl
  dev.jsonl
  valid.jsonl
```

### 3. Run a small training test

```bash
python src/train.py --train-limit 64 --dev-limit 32 --max-steps 8 --batch-size 4 --output-dir models/roberta-smoke
```

The script downloads `FacebookAI/roberta-base` if needed, runs fine-tuning and evaluation, and saves the model and tokenizer.

A CUDA-enabled GPU is recommended for training. The current implementation has been tested on an NVIDIA RTX 4080 Laptop GPU.

## Next Steps

- Run full RoBERTa fine-tuning and evaluate classification performance.
- Implement evidence masking and matched-control masking.
- Compare gold-label probability changes across experimental conditions.
- Pilot manually checked factuality-changing edits if time permits.

## References

Li et al. (2024). [MAVEN-FACT: A Large-scale Event Factuality Detection Dataset](https://aclanthology.org/2024.findings-emnlp.651/).

DeYoung et al. (2020). [ERASER: A Benchmark to Evaluate Rationalized NLP Models](https://aclanthology.org/2020.acl-main.408/).
