"""
svm_models.py: MEMBER 1 
Linear, polynomial and RBF SVM pipelines with their hyperparameter grids.
"""

from sklearn.svm import SVC

from .preprocessing import build_pipeline


def create_svm_searches():
    # Returns {model_name: (pipeline, parameter_grid)} for the three SVMs.

    # Linear SVM.
    linear_pipeline = build_pipeline(SVC(kernel="linear"))
    linear_grid = {
        "classifier__C": [0.1, 1, 10, 100],
    }

    # Polynomial SVM.
    polynomial_pipeline = build_pipeline(SVC(kernel="poly"))
    polynomial_grid = {
        "classifier__C": [0.1, 1, 10],
        "classifier__degree": [2, 3],
        "classifier__gamma": ["scale", "auto"],
        "classifier__coef0": [0.0, 1.0],
    }

    # RBF SVM.
    rbf_pipeline = build_pipeline(SVC(kernel="rbf"))
    rbf_grid = {
        "classifier__C": [0.1, 1, 10, 100],
        "classifier__gamma": ["scale", "auto", 0.01, 0.1, 1],
    }

    return {
        "SVM-Linear": (linear_pipeline, linear_grid),
        "SVM-Polynomial": (polynomial_pipeline, polynomial_grid),
        "SVM-RBF": (rbf_pipeline, rbf_grid),
    }