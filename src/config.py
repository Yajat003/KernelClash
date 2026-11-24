"""
config.py: MEMBER 3 
Central configuration: constants, paths, class names, warning filters, plot theme,
so every other module imports its settings from here.
"""

from pathlib import Path
import warnings

import seaborn as sns
from sklearn.exceptions import ConvergenceWarning

# Experiment settings
RANDOM_STATE = 42
TEST_SIZE = 0.25
CV_FOLDS = 5

# Paths (relative to the project root, so the code runs from anywhere)
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "data_banknote_authentication.txt"

OUTPUT_DIR = BASE_DIR / "outputs"
EDA_DIR = OUTPUT_DIR / "eda"                # EDA + PCA figures
MODELS_DIR = OUTPUT_DIR / "models"          # tuning results, MLP loss curve
EVALUATION_DIR = OUTPUT_DIR / "evaluation"  # metrics, reports, plots, predictions

for _directory in (EDA_DIR, MODELS_DIR, EVALUATION_DIR):
    _directory.mkdir(parents=True, exist_ok=True)

# Dataset schema
FEATURE_NAMES = [
    "Variance",
    "Skewness",
    "Curtosis",
    "Entropy",
]

TARGET_NAME = "Class"

CLASS_NAMES = {
    0: "Authentic",
    1: "Fake",
}

# Warnings and plotting theme Keep warnings visible except the convergence warning, which is captured and reported after model fitting (see model_training.py).
warnings.filterwarnings("default")
warnings.filterwarnings("ignore", category=ConvergenceWarning)

sns.set_theme(style="whitegrid", context="notebook")
