"""Feature engineering for the Titanic dataset.

Everything that "learns" something from the data (median ages, ticket group
sizes, the most common port) lives inside `TitanicFeatures.fit`, so it only
ever sees the training split. That keeps the cross-validation scores honest:
no information from the validation/test rows leaks into the features.
"""

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

TITLE_MAP = {
    "Mr": "Mr",
    "Miss": "Miss", "Mlle": "Miss", "Ms": "Miss",
    "Mrs": "Mrs", "Mme": "Mrs", "Lady": "Mrs", "the Countess": "Mrs", "Dona": "Mrs",
    "Master": "Master",
    # Small groups get merged so the model doesn't memorise 1-2 passengers.
    "Dr": "Officer", "Rev": "Officer", "Major": "Officer", "Col": "Officer",
    "Capt": "Officer", "Sir": "Noble", "Don": "Noble", "Jonkheer": "Noble",
}

NUMERIC_FEATURES = ["Age", "Fare", "FarePerPerson", "FamilySize", "TicketGroupSize", "Pclass"]
CATEGORICAL_FEATURES = ["Sex", "Title", "Embarked", "Deck", "FamilyType", "IsChild", "HasCabin"]


def extract_title(name: str) -> str:
    """'Braund, Mr. Owen Harris' -> 'Mr' (grouped via TITLE_MAP)."""
    match = re.search(r",\s*([^.]+)\.", str(name))
    raw = match.group(1).strip() if match else "Unknown"
    return TITLE_MAP.get(raw, "Rare")


def family_type(size: int) -> str:
    if size == 1:
        return "Alone"
    if size <= 4:
        return "Small"
    return "Large"


class TitanicFeatures(BaseEstimator, TransformerMixin):
    """Turns raw Titanic columns into model-ready features."""

    def fit(self, X: pd.DataFrame, y=None):
        df = self._base(X)
        # Age is missing for ~20% of passengers. "Master" (boys) and "Mr"
        # have very different typical ages, so impute by title + class.
        self.age_by_title_class_ = df.groupby(["Title", "Pclass"])["Age"].median()
        self.age_by_title_ = df.groupby("Title")["Age"].median()
        self.age_overall_ = df["Age"].median()
        self.fare_median_ = df["Fare"].median()
        self.embarked_mode_ = df["Embarked"].mode().iloc[0]
        # People on the same ticket travelled together (families, friends, staff).
        self.ticket_counts_ = df["Ticket"].value_counts()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = self._base(X)

        missing_age = df["Age"].isna()
        df.loc[missing_age, "Age"] = [
            self._impute_age(t, c) for t, c in zip(df.loc[missing_age, "Title"], df.loc[missing_age, "Pclass"])
        ]
        df["Fare"] = df["Fare"].fillna(self.fare_median_)
        df["Embarked"] = df["Embarked"].fillna(self.embarked_mode_)

        df["TicketGroupSize"] = df["Ticket"].map(self.ticket_counts_).fillna(1)
        # Never count fewer people than the family we already know about.
        df["TicketGroupSize"] = np.maximum(df["TicketGroupSize"], df["FamilySize"])
        # 'Fare' is the price of the whole ticket, not per person.
        df["FarePerPerson"] = df["Fare"] / df["TicketGroupSize"]

        df["IsChild"] = (df["Age"] < 14).map({True: "yes", False: "no"})
        return df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _base(X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df["Title"] = df["Name"].map(extract_title)
        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
        df["FamilyType"] = df["FamilySize"].map(family_type)
        df["HasCabin"] = df["Cabin"].notna().map({True: "yes", False: "no"})
        # First letter of the cabin = deck. T only has one passenger -> unknown.
        df["Deck"] = df["Cabin"].str[0].fillna("U").replace({"T": "U"})
        return df

    def _impute_age(self, title, pclass):
        if (title, pclass) in self.age_by_title_class_.index:
            value = self.age_by_title_class_[(title, pclass)]
            if not np.isnan(value):
                return value
        return self.age_by_title_.get(title, self.age_overall_)
