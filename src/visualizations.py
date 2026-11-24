"""
visualizations.py: MEMBER 1
Model-performance plots (outputs/evaluation/) and the PCA data projection 
(outputs/eda/): confusion matrices, ROC curves, metric bars, PCA scatter.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)

from .config import RANDOM_STATE, EDA_DIR, EVALUATION_DIR
from .evaluation import get_positive_class_scores


def plot_confusion_matrices(fitted_models, X_test, y_test):
    # Plot a confusion matrix for each model.

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.flatten()

    for index, (model_name, search) in enumerate(fitted_models.items()):

        y_pred = search.best_estimator_.predict(X_test)

        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1],
        )

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Authentic", "Fake"],
        )

        display.plot(
            ax=axes[index],
            cmap="Blues",
            colorbar=False,
            values_format="d",
        )

        axes[index].set_title(model_name)

    fig.suptitle("Confusion Matrix Comparison", fontsize=16)
    plt.tight_layout()
    plt.savefig(EVALUATION_DIR / "04_confusion_matrices.png", dpi=300)
    plt.show()


def plot_roc_curves(fitted_models, X_test, y_test):
    # Compare ROC curves and AUC values on the same test observations.

    plt.figure(figsize=(9, 7))

    for model_name, search in fitted_models.items():

        model = search.best_estimator_
        y_score = get_positive_class_scores(model, X_test)

        RocCurveDisplay.from_predictions(
            y_test,
            y_score,
            name=model_name,
            ax=plt.gca(),
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random classifier",
    )

    plt.title("ROC Curve Comparison")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(EVALUATION_DIR / "05_roc_curve_comparison.png", dpi=300)
    plt.show()


def plot_metric_comparison(metrics_df: pd.DataFrame):
    # Visualize the evaluation metrics of all models.

    metric_columns = [
        "Accuracy",
        "Precision (Fake)",
        "Recall (Fake)",
        "F1-score (Fake)",
        "ROC-AUC",
        "Balanced Accuracy",
    ]

    plot_data = metrics_df.melt(
        id_vars="Model",
        value_vars=metric_columns,
        var_name="Metric",
        value_name="Score",
    )

    plt.figure(figsize=(14, 7))

    sns.barplot(
        data=plot_data,
        x="Metric",
        y="Score",
        hue="Model",
    )

    plt.ylim(0, 1.05)
    plt.title("Performance Metric Comparison")
    plt.xlabel("Evaluation Metric")
    plt.ylabel("Score")
    plt.xticks(rotation=20, ha="right")
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(EVALUATION_DIR / "06_metric_comparison.png", dpi=300)
    plt.show()


def plot_pca_visualization(X_train, y_train):
    # Show a two-dimensional PCA projection of the scaled training data. PCA is used only for visualization. 
    # The original four features are still used for model training and check.

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(
        {
            "Principal Component 1": X_pca[:, 0],
            "Principal Component 2": X_pca[:, 1],
            "Class": y_train.to_numpy(),
        }
    )

    plt.figure(figsize=(9, 7))

    sns.scatterplot(
        data=pca_df,
        x="Principal Component 1",
        y="Principal Component 2",
        hue="Class",
        palette="Set1",
        alpha=0.7,
    )

    explained_variance = pca.explained_variance_ratio_.sum() * 100

    plt.title(
        "PCA Projection of Banknote Data "
        f"(First Two Components: {explained_variance:.2f}% Variance)"
    )
    plt.tight_layout()
    plt.savefig(EDA_DIR / "08_pca_visualization.png", dpi=300)
    plt.show()

    print(
        "\nPCA explained variance by first two components: "
        f"{explained_variance:.2f}%"
    )
    print(
        "Note: PCA is used only to visualize the four-dimensional data "
        "in two dimensions. It is not used as the model input."
    )