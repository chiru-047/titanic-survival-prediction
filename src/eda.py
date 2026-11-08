"""Exploratory data analysis: saves the charts used in the README.

Run:  python src/eda.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from features import TitanicFeatures

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "reports" / "figures"


def survival_rate(df, col):
    return df.groupby(col, observed=True)["Survived"].mean().mul(100)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(ROOT / "data" / "train.csv")
    feats = TitanicFeatures().fit(raw).transform(raw)
    df = feats.assign(Survived=raw["Survived"])

    print(f"Passengers: {len(raw)} | survived: {raw.Survived.mean():.1%}")
    print("Missing values:\n", raw.isna().sum()[raw.isna().sum() > 0], "\n")

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    panels = [
        ("Sex", "Sex"),
        ("Pclass", "Ticket class"),
        ("Title", "Title (from name)"),
        ("FamilyType", "Family size"),
        ("Deck", "Deck (from cabin)"),
        ("Embarked", "Port of embarkation"),
    ]
    for ax, (col, label) in zip(axes.flat, panels):
        rate = survival_rate(df, col)
        counts = df[col].value_counts()
        ax.bar(rate.index.astype(str), rate.values, color="#4C72B0")
        ax.axhline(raw.Survived.mean() * 100, ls="--", color="grey", lw=1)
        ax.set_title(label)
        ax.set_ylabel("Survival rate (%)")
        ax.set_ylim(0, 105)
        for i, (k, v) in enumerate(rate.items()):
            ax.text(i, v + 2, f"n={counts[k]}", ha="center", fontsize=8)
        print(f"{label}:\n{rate.round(1).to_string()}\n")
    fig.suptitle("Survival rate by group (dashed line = overall 38%)")
    fig.tight_layout()
    fig.savefig(FIGURES / "survival_by_group.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.6))
    bins = range(0, 85, 5)
    ax.hist(df.loc[df.Survived == 0, "Age"], bins=bins, alpha=0.6, label="Died", color="#C44E52")
    ax.hist(df.loc[df.Survived == 1, "Age"], bins=bins, alpha=0.6, label="Survived", color="#55A868")
    ax.set_xlabel("Age (missing ages imputed by title + class)")
    ax.set_ylabel("Passengers")
    ax.set_title("Age distribution — children were more likely to survive")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "age_distribution.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
