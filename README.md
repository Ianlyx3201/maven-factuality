
# Evidence Sensitivity in Neural Event Factuality Prediction

LING 482 Term Project, Fall 2026

## Research Question

Does a neural event factuality classifier rely more on human-annotated factuality evidence than on comparable unannotated text?

The main hypothesis is that masking annotated evidence will cause a larger decrease in gold-label probability than masking a length-matched control span.

## Dataset

This project uses [MAVEN-FACT](https://github.com/THU-KEG/MAVEN-FACT) (Li et al., 2024), which provides event factuality labels and supporting-word annotations.

Download `train.jsonl` and `valid.jsonl` from the dataset repository's [Google Drive folder](https://drive.google.com/drive/folders/1GP8cdPlcUx5Om-lQse11LTDU-QFeAMom).

Place both files in a local `data/` directory:

```text
data/
  train.jsonl
  valid.jsonl
```

Dataset files are excluded from Git.

## Methods

1. Inspect event labels and supporting-evidence annotations.
2. Fine-tune RoBERTa-base for five-class event factuality prediction.
3. Compare predictions on original inputs, evidence-masked inputs, and matched-control-masked inputs.
4. Measure changes in gold-label probability, alongside macro-F1 and per-class classification performance.
5. If feasible, evaluate manually checked factuality-changing sentence edits.

## Repository Structure

```text
src/
  inspect_data.py    Dataset statistics and evidence-alignment checks
  preprocess.py      Model input preparation (planned)
  train.py           RoBERTa fine-tuning (planned)
  evaluate.py        Classification evaluation (planned)
  masking.py         Evidence and control masking experiments (planned)
```

Additional scripts and experiment configurations will be added as implementation proceeds.

## Current Progress

The dataset inspection script is implemented and tested on the original training and validation files.

| Statistic | Train | Validation |
|---|---:|---:|
| Documents | 2,913 | 710 |
| Event mentions | 73,939 | 17,780 |
| Mentions with evidence | 3,724 | 843 |
| Evidence passing initial alignment checks | 3,584 | 797 |

These counts reflect preliminary annotation checks, not the final experimental sample. Model training and masking experiments are not yet implemented.

## Running the Inspection Script

Requires Python 3.10 or later. No additional packages are needed for the current script.

From the repository root, run:

```bash
python src/inspect_data.py
```

The script reports document counts, factuality-label distributions, and evidence-alignment statistics.

## References

Li et al. (2024). [MAVEN-FACT: A Large-scale Event Factuality Detection Dataset](https://aclanthology.org/2024.findings-emnlp.651/).

DeYoung et al. (2020). [ERASER: A Benchmark to Evaluate Rationalized NLP Models](https://aclanthology.org/2020.acl-main.408/).
