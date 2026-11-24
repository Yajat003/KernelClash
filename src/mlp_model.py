"""
The architecture / convergence analysis (saved to outputs/models/), 
so mlp_model.py, MEMBER 2 Two-hidden-layer MLP (backpropagation): pipeline + grid.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

from .config import FEATURE_NAMES, RANDOM_STATE, MODELS_DIR
from .preprocessing import build_pipeline


def create_mlp_search():
    # Returns {model_name: (pipeline, parameter_grid)} for the MLP.

    # Multilayer Perceptron with two hidden layers.
    mlp_pipeline = build_pipeline(
        MLPClassifier(
            hidden_layer_sizes=(32, 16),
            max_iter=2000,
            random_state=RANDOM_STATE,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=30,
            tol=1e-4,
        )
    )

    mlp_grid = {
        "classifier__hidden_layer_sizes": [
            (32, 16),
            (64, 32),
            (50, 25),
        ],
        "classifier__activation": ["relu", "tanh"],
        "classifier__alpha": [0.0001, 0.001, 0.01],
        "classifier__learning_rate_init": [0.001, 0.01],
    }

    return {
        "MLP-Backpropagation": (mlp_pipeline, mlp_grid),
    }


def analyze_mlp(fitted_models):
    # The best MLP is selected using cross-validation F1, so print the selected MLP architecture and plot its loss curve.

    mlp_search = fitted_models["MLP-Backpropagation"]
    mlp_pipeline = mlp_search.best_estimator_
    mlp = mlp_pipeline.named_steps["classifier"]

    print("\n" + "=" * 75)
    print("MLP ARCHITECTURE AND CONVERGENCE ANALYSIS")
    print("=" * 75)

    print("\nSelected MLP parameters:")
    for parameter, value in mlp_search.best_params_.items():
        print(f"{parameter}: {value}")

    print("\nMLP architecture:")
    print(f"Input layer:  {len(FEATURE_NAMES)} neurons")
    print(f"Hidden layer 1: {mlp.hidden_layer_sizes[0]} neurons")
    print(f"Hidden layer 2: {mlp.hidden_layer_sizes[1]} neurons")
    print("Output layer: 1 binary classification output")
    print(f"Activation function: {mlp.activation}")
    print(f"Optimizer: {mlp.solver}")
    print(f"Iterations performed: {mlp.n_iter_}")
    print(f"Maximum iterations: {mlp.max_iter}")
    print(f"Final training loss: {mlp.loss_:.6f}")
    print(f"Early stopping enabled: {mlp.early_stopping}")

    if hasattr(mlp, "best_validation_score_"):
        print(
            "Best internal validation accuracy: "
            f"{mlp.best_validation_score_:.4f}"
        )

    if hasattr(mlp, "loss_curve_") and len(mlp.loss_curve_) > 0:

        plt.figure(figsize=(9, 5))

        plt.plot(
            range(1, len(mlp.loss_curve_) + 1),
            mlp.loss_curve_,
            label="Training loss",
        )

        plt.title("MLP Training Loss Curve")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(MODELS_DIR / "07_mlp_convergence_curve.png", dpi=300)
        plt.show()

        loss_decreased = mlp.loss_curve_[-1] < mlp.loss_curve_[0]

        print("\nConvergence observations:")
        print(f"Initial loss: {mlp.loss_curve_[0]:.6f}")
        print(f"Final loss:   {mlp.loss_curve_[-1]:.6f}")
        print(f"Loss decreased overall: {loss_decreased}")

        if mlp.n_iter_ >= mlp.max_iter:
            print(
                "The MLP reached the configured maximum iteration count. "
                "Check whether it converged before the limit."
            )
        else:
            print(
                "The MLP stopped before reaching the maximum iteration "
                "count, subject to its stopping criteria."
            )

        loss_df = pd.DataFrame(
            {
                "Iteration": range(1, len(mlp.loss_curve_) + 1),
                "Training Loss": mlp.loss_curve_,
            }
        )
        loss_df.to_csv(
            MODELS_DIR / "mlp_training_loss.csv",
            index=False,
        )

    else:
        print("The fitted MLP does not expose a usable loss curve.")
