"""
data_loader.py: MEMBER 3 
Loads the headerless Banknote Authentication file and checks it.
"""

from pathlib import Path

import pandas as pd

from .config import FEATURE_NAMES, TARGET_NAME


def load_dataset(file_path: Path) -> pd.DataFrame:
    # Load the headerless Banknote Authentication text file.

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path.resolve()}\n"
            "Place data_banknote_authentication.txt in the data/ folder, "
            "or update DATA_PATH in src/config.py."
        )

    data = pd.read_csv(
        file_path,
        header=None,
        names=FEATURE_NAMES + [TARGET_NAME],
        sep=",",
        skipinitialspace=True,
    )

    if data.shape[1] != 5:
        raise ValueError(
            f"Expected 5 columns (4 features + target), "
            f"but found {data.shape[1]}."
        )

    # Convert all columns to numeric and fail clearly if invalid values exist.
    for column in FEATURE_NAMES + [TARGET_NAME]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    if data.isnull().any().any():
        missing = data.isnull().sum()
        raise ValueError(
            "Missing or non-numeric values were found:\n"
            f"{missing[missing > 0]}"
        )

    # Only the two projected class labels must be contained by the target.
    observed_classes = set(data[TARGET_NAME].unique())
    if observed_classes != {0, 1}:
        raise ValueError(
            f"Expected target labels {{0, 1}}, found {observed_classes}."
        )

    data[TARGET_NAME] = data[TARGET_NAME].astype(int)

    return data
