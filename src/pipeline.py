"""Builds full sklearn pipelines: feature engineering -> encoding -> model."""

from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TitanicFeatures

RANDOM_STATE = 42


def candidate_models() -> dict:
    """Models compared in train.py. Kept deliberately simple and explainable."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=0.5),
        "K-Nearest Neighbours": KNeighborsClassifier(n_neighbors=11),
        # SVMs don't output probabilities natively, so calibrate them.
        "SVM (RBF kernel)": CalibratedClassifierCV(SVC(C=1.0), ensemble=False),
        "Random Forest": RandomForestClassifier(
            n_estimators=400, max_depth=7, min_samples_leaf=2, random_state=RANDOM_STATE
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=RANDOM_STATE
        ),
    }


def build_pipeline(model) -> Pipeline:
    encode = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline([("features", TitanicFeatures()), ("encode", encode), ("model", model)])


def ensemble_model() -> VotingClassifier:
    """Soft voting = average the predicted probabilities of 4 different model types.

    Their mistakes are partly independent, so the average is more stable than
    any single model (KNN is left out because it was clearly the weakest).
    """
    members = {k: v for k, v in candidate_models().items() if k != "K-Nearest Neighbours"}
    short = {"Logistic Regression": "logreg", "SVM (RBF kernel)": "svm",
             "Random Forest": "rf", "Gradient Boosting": "gb"}
    return VotingClassifier([(short[k], m) for k, m in members.items()], voting="soft")
