"""
evaluation.py: MEMBER 1
Scores every tuned model on the untouched test set and saves the metric table, 
classification reports and test-set predictions (outputs/evaluation/).
"""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
    classification_report,
)

from .config import CLASS_NAMES, EVALUATION_DIR


def get_positive_class_scores(model, X):
    # Return continuous scores for class 1. SVC uses decision_function because probability=True isn't needed. MLP uses predict_proba.

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)

        # Sklearn's binary decision function is oriented toward the classes_[1] for binary category. Confirm that class 1 is classes_[1].
        classes = model.classes_
        if classes[1] != 1:
            scores = -scores

        return scores

    probabilities = model.predict_proba(X)
    positive_index = list(model.classes_).index(1)

    return probabilities[:, positive_index]


def evaluate_models(fitted_models, X_test, y_test):
    # On the same test set, evaluate all tuned models.

    metrics_records = []
    predictions = {}
    scores = {}
    reports = {}

    print("\n" + "=" * 75)
    print("FINAL TEST-SET EVALUATION")
    print("=" * 75)

    for model_name, search in fitted_models.items():

        best_model = search.best_estimator_

        y_pred = best_model.predict(X_test)
        y_score = get_positive_class_scores(best_model, X_test)

        predictions[model_name] = y_pred
        scores[model_name] = y_score

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        auc = roc_auc_score(y_test, y_score)
        balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

        metrics_records.append(
            {
                "Model": model_name,
                "Accuracy": accuracy,
                "Precision (Fake)": precision,
                "Recall (Fake)": recall,
                "F1-score (Fake)": f1,
                "ROC-AUC": auc,
                "Balanced Accuracy": balanced_accuracy,
            }
        )

        report = classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["Authentic (0)", "Fake (1)"],
            digits=4,
            zero_division=0,
        )
        reports[model_name] = report

        print(f"\n{model_name}")
        print("-" * 60)
        print(f"Accuracy:          {accuracy:.4f}")
        print(f"Precision (Fake):  {precision:.4f}")
        print(f"Recall (Fake):     {recall:.4f}")
        print(f"F1-score (Fake):   {f1:.4f}")
        print(f"ROC-AUC:           {auc:.4f}")
        print(f"Balanced Accuracy: {balanced_accuracy:.4f}")

        print("\nClassification report:")
        print(report)

    metrics_df = pd.DataFrame(metrics_records)
    metrics_df = metrics_df.sort_values(
        by="F1-score (Fake)",
        ascending=False,
    )

    metrics_df.to_csv(
        EVALUATION_DIR / "model_performance_metrics.csv",
        index=False,
    )

    with open(
        EVALUATION_DIR / "classification_reports.txt",
        "w",
        encoding="utf-8",
    ) as file:
        for model_name, report in reports.items():
            file.write(f"\n{model_name}\n")
            file.write("=" * 70 + "\n")
            file.write(report + "\n")

    print("\n" + "=" * 75)
    print("MODEL COMPARISON TABLE")
    print("=" * 75)
    print(metrics_df.round(4).to_string(index=False))

    return metrics_df, predictions, scores


def save_test_predictions(fitted_models, X_test, y_test):
    # Save actual labels and predictions from all models.

    prediction_df = pd.DataFrame(
        {
            "Actual Class": y_test.to_numpy(),
            "Actual Label": [
                CLASS_NAMES[value] for value in y_test.to_numpy()
            ],
        },
        index=X_test.index,
    )

    for model_name, search in fitted_models.items():
        prediction_df[f"{model_name} Prediction"] = (
            search.best_estimator_.predict(X_test)
        )

    prediction_df.to_csv(
        EVALUATION_DIR / "test_set_predictions.csv",
        index=True,
        index_label="Original Row Index",
    )