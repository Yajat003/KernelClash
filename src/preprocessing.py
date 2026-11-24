"""
preprocessing.py: MEMBER 3 
Stratified train/test split and the leakage-safe scaling pipeline builder.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import FEATURE_NAMES, TARGET_NAME, TEST_SIZE, RANDOM_STATE


def prepare_data(data: pd.DataFrame):
    # Separate features and target, then perform a stratified split.

    X = data[FEATURE_NAMES]
    y = data[TARGET_NAME]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\n" + "=" * 75)
    print("TRAIN-TEST SPLIT")
    print("=" * 75)

    print(f"Training observations: {len(X_train)}")
    print(f"Testing observations:  {len(X_test)}")

    print("\nTraining class distribution:")
    print(y_train.value_counts().sort_index())

    print("\nTesting class distribution:")
    print(y_test.value_counts().sort_index())

    return X_train, X_test, y_train, y_test


def build_pipeline(classifier) -> Pipeline:
    # StandardScaler is, of course, inside every pipeline. During cross-validation, the scaler is fitted only on the corresponding training fold (no leakage).
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )
