"""Train and evaluate Titanic survival models.

Steps
1. Hold out 20% of passengers as a final test set (never touched until the end).
2. Compare 5 models + a soft-voting ensemble with 5x repeated stratified
   5-fold cross-validation on the other 80%.
3. Report accuracy / precision / recall / F1 / ROC-AUC on the held-out set.
4. Save the model, metrics and figures.

Why no big grid search? An early version tuned the SVM over 12 combinations;
it reached 84% CV accuracy but only 77% on the test set -- with ~700 training
rows, heavy tuning just fits the noise in the folds. All models land within
~1% of each other, so the ensemble (most stable across folds) is shipped.

Run:  python src/train.py
"""

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split

from pipeline import RANDOM_STATE, build_pipeline, candidate_models, ensemble_model

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "train.csv"
FIGURES = ROOT / "reports" / "figures"
MODELS = ROOT / "models"
FINAL_MODEL = "Soft-voting ensemble"


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA)
    X, y = df.drop(columns=["Survived"]), df["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    # Repeating 5-fold CV 5 times (25 scores per model) gives a much less noisy
    # estimate than a single 5-fold run on a dataset this small.
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_STATE)

    # ---- 1. compare models -------------------------------------------------
    print(f"Baseline (predict everyone dies): {1 - y_train.mean():.3f}\n")
    models = {**candidate_models(), FINAL_MODEL: ensemble_model()}
    rows = []
    for name, model in models.items():
        scores = cross_val_score(build_pipeline(model), X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
        rows.append({"model": name, "cv_accuracy_mean": round(scores.mean(), 4),
                     "cv_accuracy_std": round(scores.std(), 4)})
        print(f"{name:<22} CV accuracy {scores.mean():.3f} ± {scores.std():.3f}")
    comparison = pd.DataFrame(rows).sort_values("cv_accuracy_mean", ascending=False)
    comparison.to_csv(ROOT / "reports" / "model_comparison.csv", index=False)
    plot_comparison(comparison)

    # ---- 2. final evaluation on the untouched test set ---------------------
    best = build_pipeline(models[FINAL_MODEL]).fit(X_train, y_train)
    pred = best.predict(X_test)
    proba = best.predict_proba(X_test)[:, 1]
    metrics = {
        "model": FINAL_MODEL,
        "cv_accuracy": float(comparison.set_index("model").loc[FINAL_MODEL, "cv_accuracy_mean"]),
        "test_accuracy": round(accuracy_score(y_test, pred), 4),
        "test_precision": round(precision_score(y_test, pred), 4),
        "test_recall": round(recall_score(y_test, pred), 4),
        "test_f1": round(f1_score(y_test, pred), 4),
        "test_roc_auc": round(roc_auc_score(y_test, proba), 4),
        "baseline_accuracy": round(1 - y_test.mean(), 4),
    }
    print(f"\nHeld-out test set ({FINAL_MODEL}):")
    for k, v in metrics.items():
        if k.startswith("test_") or k == "baseline_accuracy":
            print(f"  {k:<18} {v}")
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(metrics, indent=2))

    plot_evaluation(best, X_test, y_test)
    plot_importance(best, X_test, y_test)

    # Refit on all labelled data before saving, so the shipped model uses every row.
    final = clone(best).fit(X, y)
    joblib.dump(final, MODELS / "titanic_model.joblib")
    print(f"\nSaved model -> {MODELS / 'titanic_model.joblib'}")


# ---------------------------------------------------------------- plotting
def plot_comparison(comparison: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    c = comparison.sort_values("cv_accuracy_mean")
    ax.barh(c["model"], c["cv_accuracy_mean"], xerr=c["cv_accuracy_std"], color="#4C72B0", capsize=3)
    ax.set_xlim(0.7, 0.9)
    ax.set_xlabel("5-fold CV accuracy")
    ax.set_title("Model comparison")
    for i, v in enumerate(c["cv_accuracy_mean"]):
        ax.text(v + 0.012, i, f"{v:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "model_comparison.png", dpi=150)
    plt.close(fig)


def plot_evaluation(model, X_test, y_test):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, display_labels=["Died", "Survived"], cmap="Blues", ax=axes[0], colorbar=False
    )
    axes[0].set_title("Confusion matrix (test set)")
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=axes[1])
    axes[1].plot([0, 1], [0, 1], "--", color="grey")
    axes[1].set_title("ROC curve (test set)")
    fig.tight_layout()
    fig.savefig(FIGURES / "evaluation.png", dpi=150)
    plt.close(fig)


def plot_importance(model, X_test, y_test):
    # Permutation importance works for any model and is measured on raw columns,
    # so it answers "how much worse does the model get if this column is shuffled?"
    result = permutation_importance(model, X_test, y_test, n_repeats=20, random_state=RANDOM_STATE)
    imp = pd.Series(result.importances_mean, index=X_test.columns)
    imp = imp[imp.abs() > 1e-4].sort_values()
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.barh(imp.index, imp.values, color=np.where(imp.values > 0, "#55A868", "#C44E52"))
    ax.set_xlabel("Drop in accuracy when column is shuffled")
    ax.set_title("Which raw columns matter most")
    fig.tight_layout()
    fig.savefig(FIGURES / "feature_importance.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
