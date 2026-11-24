"""
model_training.py: MEMBER 2 
Combines the SVM and MLP search definitions and tunes every model with 
stratified 5-fold cross-validation (GridSearchCV, scoring = F1).
"""

import warnings

import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from .config import CV_FOLDS, RANDOM_STATE, MODELS_DIR
from .svm_models import create_svm_searches
from .mlp_model import create_mlp_search


def create_model_searches():
    # SVM-Linear, SVM-Polynomial, SVM-RBF, then MLP-Backpropagation.
    return {**create_svm_searches(), **create_mlp_search()}


def tune_models(X_train, y_train):
    # Tune each model using stratified cross-validation.

    model_searches = create_model_searches()

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fitted_models = {}
    tuning_records = []

    print("\n" + "=" * 75)
    print("HYPERPARAMETER TUNING")
    print("=" * 75)

    for model_name, (pipeline, parameter_grid) in model_searches.items():

        print(f"\nTraining and tuning: {model_name}")

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=parameter_grid,
            scoring="f1",
            cv=cv,
            n_jobs=-1,
            refit=True,
            verbose=1,
            return_train_score=True,
        )

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", ConvergenceWarning)
            search.fit(X_train, y_train)

        convergence_messages = [
            str(warning.message)
            for warning in caught_warnings
            if issubclass(warning.category, ConvergenceWarning)
        ]

        fitted_models[model_name] = search

        print(f"\nBest cross-validation F1: {search.best_score_:.4f}")
        print("Best parameters:")
        for parameter, value in search.best_params_.items():
            print(f"  {parameter}: {value}")

        if convergence_messages:
            print(
                f"Convergence warnings during tuning: "
                f"{len(convergence_messages)}"
            )
        else:
            print("No convergence warnings were captured during tuning.")

        tuning_records.append(
            {
                "Model": model_name,
                "Best CV F1": search.best_score_,
                "Best Parameters": str(search.best_params_),
                "Convergence Warnings": len(convergence_messages),
            }
        )

    tuning_results = pd.DataFrame(tuning_records)
    tuning_results.to_csv(
        MODELS_DIR / "hyperparameter_tuning_results.csv",
        index=False,
    )

    print("\nHyperparameter tuning summary:")
    print(tuning_results.to_string(index=False))

    return fitted_models, tuning_results
