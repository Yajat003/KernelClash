"""
main.py: MEMBER 5
the full pipeline. 
Usage (from the project root):  python main.py
"""

from src.config import DATA_PATH, OUTPUT_DIR
from src.data_loader import load_dataset
from src.eda import perform_eda
from src.preprocessing import prepare_data
from src.model_training import tune_models
from src.evaluation import evaluate_models, save_test_predictions
from src.visualizations import (
    plot_confusion_matrices,
    plot_roc_curves,
    plot_metric_comparison,
    plot_pca_visualization,
)
from src.mlp_model import analyze_mlp


def main():

    print("=" * 75)
    print("ASSIGNMENT 7: SVM KERNELS AND NEURAL NETWORKS")
    print("UCI BANKNOTE AUTHENTICATION DATASET")
    print("=" * 75)

    # Step 1: Load data.
    data = load_dataset(DATA_PATH)

    # Step 2: Exploratory data analysis.
    perform_eda(data)

    # Step 3: Split data.
    X_train, X_test, y_train, y_test = prepare_data(data)

    # Step 4: Tune models.
    fitted_models, tuning_results = tune_models(X_train, y_train)

    # On the same test set, step 5: Evaluate all models.
    metrics_df, predictions, scores = evaluate_models(
        fitted_models,
        X_test,
        y_test,
    )

    # Step 6: Visualize model performance.
    plot_confusion_matrices(fitted_models, X_test, y_test)
    plot_roc_curves(fitted_models, X_test, y_test)
    plot_metric_comparison(metrics_df)

    # Step 7: Inspect MLP architecture and convergence.
    analyze_mlp(fitted_models)

    # Step 8: Visualize the dataset using PCA.
    plot_pca_visualization(X_train, y_train)

    # Step 9: Save model predictions.
    save_test_predictions(fitted_models, X_test, y_test)

    # Step 10: Identify the best model according to test F1.
    best_row = metrics_df.iloc[0]

    print("\n" + "=" * 75)
    print("FINAL SUMMARY")
    print("=" * 75)

    print(f"Best model by test F1-score: {best_row['Model']}")
    print(f"Test F1-score: {best_row['F1-score (Fake)']:.4f}")
    print(f"Test accuracy: {best_row['Accuracy']:.4f}")
    print(f"Fake-banknote recall: {best_row['Recall (Fake)']:.4f}")
    print(f"ROC-AUC: {best_row['ROC-AUC']:.4f}")

    print(f"\nAll output files have been saved to: {OUTPUT_DIR.resolve()}")

    print("\nGenerated files:")
    for output_file in sorted(OUTPUT_DIR.rglob("*")):
        if output_file.is_file():
            print(f"  - {output_file.relative_to(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
