"""
eda.py: MEMBER 4 
Exploratory data analysis: console summary + 3 saved figures (outputs/eda/).
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .config import FEATURE_NAMES, TARGET_NAME, CLASS_NAMES, EDA_DIR


def perform_eda(data: pd.DataFrame) -> None:
    # Display and save basic dataset information and visualizations.

    print("\n" + "=" * 75)
    print("DATASET OVERVIEW")
    print("=" * 75)

    print(f"\nDataset shape: {data.shape}")
    print(f"Number of observations: {data.shape[0]}")
    print(f"Number of input features: {len(FEATURE_NAMES)}")

    print("\nFirst five observations:")
    print(data.head())

    print("\nData types:")
    print(data.dtypes)

    print("\nDescriptive statistics:")
    print(data.describe().round(4))

    print("\nMissing values:")
    print(data.isnull().sum())

    print("\nDuplicate rows:")
    print(data.duplicated().sum())

    class_counts = data[TARGET_NAME].value_counts().sort_index()
    class_percentages = (
        data[TARGET_NAME].value_counts(normalize=True).sort_index() * 100
    )

    print("\nClass distribution:")
    for class_label, count in class_counts.items():
        print(
            f"Class {class_label} ({CLASS_NAMES[class_label]}): "
            f"{count} observations ({class_percentages[class_label]:.2f}%)"
        )

    # Visualization 1: class distribution.
    plt.figure(figsize=(7, 5))
    ax = sns.countplot(
        data=data,
        x=TARGET_NAME,
        hue=TARGET_NAME,
        palette="Set2",
        legend=False,
    )
    ax.set_title("Distribution of Authentic and Fake Banknotes")
    ax.set_xlabel("Banknote Class")
    ax.set_ylabel("Number of Observations")
    ax.set_xticks([0, 1], ["Authentic (0)", "Fake (1)"])

    for container in ax.containers:
        ax.bar_label(container)

    plt.tight_layout()
    plt.savefig(EDA_DIR / "01_class_distribution.png", dpi=300)
    plt.show()

    # Visualization 2: feature distributions by class.
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    for index, feature in enumerate(FEATURE_NAMES):
        sns.histplot(
            data=data,
            x=feature,
            hue=TARGET_NAME,
            kde=True,
            stat="density",
            common_norm=False,
            element="step",
            ax=axes[index],
        )
        axes[index].set_title(f"Distribution of {feature}")

    fig.suptitle(
        "Feature Distributions by Banknote Class",
        fontsize=15,
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(EDA_DIR / "02_feature_distributions.png", dpi=300)
    plt.show()

    # Visualization 3: correlation heatmap.
    plt.figure(figsize=(9, 7))
    correlation = data.corr(numeric_only=True)

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        square=True,
    )

    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "03_correlation_heatmap.png", dpi=300)
    plt.show()
