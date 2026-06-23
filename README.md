# Machine-Generated Code Detection --- SemEval 2025 ---

Telling human-written code apart from AI-generated code. This is my submission for **SemEval 2025** on Kaggle: given a source-code snippet, classify it as **human-written** (label `0`) or **machine-generated** (label `1`).

We implemented seven architectures - from classical ML baselines to deep-learning sequence models - and trained each on the full 500,000-sample dataset. The best performer, a BiLSTM, reached **96.3% accuracy** on the held-out validation set.

## Results

Evaluated on the 100,000-sample validation set, ordered by accuracy.

| Model | Type | Accuracy | Macro F1 | Train Time |
|---|---|---|---|---|
| **BiLSTM** | Deep Learning | **96.29%** | **0.9628** | 14.6 min (GPU) |
| TextCNN | Deep Learning | 96.13% | 0.9613 | 64.0 min (GPU) |
| Random Forest | Ensemble ML | 91.37% | 0.9137 | 97.0 min (CPU) |
| SVM | Classical ML | 89.51% | 0.8950 | 2.8 min (CPU) |
| Logistic Regression | Classical ML | 88.63% | 0.8863 | 0.6 min (CPU) |
| Naive Bayes | Classical ML | 81.30% | 0.8129 | 2.1 sec (CPU) |
| CodeBERT | Transformer | — | — | Not completed |

A few takeaways:

- The two deep-learning models clearly pulled ahead of every classical baseline (~96% vs. ≤91%).
- BiLSTM beat TextCNN on accuracy while training in roughly a quarter of the time.
- For a fast baseline, Logistic Regression is hard to argue with: 88.6% accuracy in under a minute on CPU.

Confusion-matrix plots for every model are in `results/plots/`.

## Models

The IDs below map to the `--model` flag used by `train.py`.

| ID | Model | Type |
|---|---|---|
| A | Logistic Regression | Classical ML |
| B | BiLSTM | Deep Learning |
| C | CodeBERT | Transformer |
| D | Naive Bayes | Classical ML |
| E | Random Forest | Ensemble ML |
| F | SVM | Classical ML |
| G | TextCNN | Deep Learning |

## Dataset

| Property | Details |
|---|---|
| Competition | SemEval 2025 Task 13 (Kaggle) |
| Training set | 500,000 samples |
| Validation set | 100,000 samples |
| Class balance | ~47.7% human / ~52.3% machine |

The `.parquet` data files are **not** tracked in this repo. Download them from Kaggle and drop them into `data/` before running anything:

https://www.kaggle.com/datasets/daniilor/semeval-2026-task13?resource=download

## Project Structure

```
├── baseline/
│   ├── model_baseline_A.py         # Logistic Regression
│   ├── model_baseline_B.py         # BiLSTM
│   ├── model_baseline_NB.py        # Naive Bayes
│   ├── model_baseline_RF.py        # Random Forest
│   ├── model_baseline_svm.py       # SVM
│   ├── model_baseline_textcnn.py   # TextCNN
│   └── model_baseline_codebert.py  # CodeBERT
├── utils/
│   ├── data_loader.py
│   ├── preprocess.py
│   └── evaluate.py
├── data/                           # Not tracked (see .gitignore)
├── results/                        # JSON reports and confusion-matrix plots
├── train.py
└── check_data.py
```

## Setup

```bash
pip install torch transformers scikit-learn pandas numpy matplotlib seaborn pyarrow tqdm joblib
```

## Usage

Pick models by ID. Results are written to `results/`.

```bash
# Classical ML models
python train.py --model A,D,E,F

# Deep-learning models (GPU Required)
python train.py --model B,G

# A single model
python train.py --model B
```
