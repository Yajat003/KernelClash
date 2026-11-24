# SVM Kernels and MLP (Backpropagation) – Banknote Authentication

Compares Linear, Polynomial and RBF SVMs with a two-hidden-layer MLP on the
UCI Banknote Authentication dataset (4 attributes, class 0 = authentic, 1 = fake).

## Setup
```bash
pip install -r requirements.txt
```
Place `data_banknote_authentication.txt` in `data/`, then run from the project root:
```bash
python main.py
```

## Pipeline
And load & validate -> EDA -> stratified 75/25 split -> scaling inside pipelines
(no leakage) -> 5-fold CV grid search (F1) for 3 SVMs + MLP -> check on the
untouched test set -> plots, reports, predictions.

## Outputs
| Folder | Contents |
|---|---|
| `outputs/eda/` | class distribution, feature distributions, correlation heatmap, PCA projection |
| `outputs/models/` | `hyperparameter_tuning_results.csv`, MLP loss curve (png + csv) |
| `outputs/evaluation/` | metrics table, classification reports, confusion matrices, ROC curves, metric comparison, test predictions |

## Team contributions
| Member | Load | Files |
|---|---|---|
| 1 | Large |TBD|
| 2 | Large | TBD |
| 3 | Small | TBD |
| 4 | Small | TBD |
| 5 | Small | TBD |
