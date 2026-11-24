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
| Member | Files | Responsibility |
|---|---|---|
| **M1 (Yajat)** | `src/evaluation.py`, `src/visualizations.py`, `src/svm_models.py` | Linear/Polynomial/RBF SVM pipelines and grids; test-set metrics, classification reports and saved predictions; confusion-matrix, ROC, metric-comparison and PCA plots |
| **M2 (Aditya)** | `src/mlp_model.py`, `src/model_training.py`, `src/__init__.py` | MLP pipeline, grid and architecture/convergence analysis; cross-validated tuning of all models with convergence-warning tracking; package setup |
| **M3 (Sai)** | `src/data_loader.py`, `src/preprocessing.py` | Dataset loading and validation; stratified split and leakage-safe scaling pipeline builder |
| **M4 (Varun)** | `src/eda.py`, `src/config.py` | Exploratory data analysis and figures; project-wide constants, paths, class names, warning filters and plot theme |
| **M5 (Abhishek)** | `main.py`, `README.md` (and `requirements.txt`) | Pipeline orchestration, final summary and output listing; project documentation and dependency list |