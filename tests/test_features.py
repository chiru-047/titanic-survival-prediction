import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TitanicFeatures, extract_title  # noqa: E402


def sample():
    return pd.DataFrame([
        {"Pclass": 3, "Name": "Braund, Mr. Owen Harris", "Sex": "male", "Age": 22, "SibSp": 1,
         "Parch": 0, "Ticket": "A/5 21171", "Fare": 7.25, "Cabin": None, "Embarked": "S"},
        {"Pclass": 1, "Name": "Cumings, Mrs. John Bradley", "Sex": "female", "Age": 38, "SibSp": 1,
         "Parch": 0, "Ticket": "PC 17599", "Fare": 71.28, "Cabin": "C85", "Embarked": "C"},
        {"Pclass": 3, "Name": "Palsson, Master. Gosta Leonard", "Sex": "male", "Age": None, "SibSp": 3,
         "Parch": 1, "Ticket": "349909", "Fare": 21.07, "Cabin": None, "Embarked": None},
        {"Pclass": 3, "Name": "Rice, Master. Eugene", "Sex": "male", "Age": 2, "SibSp": 4,
         "Parch": 1, "Ticket": "382652", "Fare": 29.12, "Cabin": None, "Embarked": "Q"},
    ])


def test_extract_title_groups_rare_titles():
    assert extract_title("Braund, Mr. Owen Harris") == "Mr"
    assert extract_title("Smith, Mlle. Anne") == "Miss"
    assert extract_title("Rothes, the Countess. of (Lucy)") == "Mrs"
    assert extract_title("Weir, Col. John") == "Officer"
    assert extract_title("no title here") == "Rare"


def test_output_columns_and_no_missing_values():
    out = TitanicFeatures().fit(sample()).transform(sample())
    assert list(out.columns) == NUMERIC_FEATURES + CATEGORICAL_FEATURES
    assert not out.isna().any().any()


def test_missing_age_imputed_from_same_title():
    out = TitanicFeatures().fit(sample()).transform(sample())
    # The only other "Master" is 2 years old, so the missing boy gets a child's age, not an adult median.
    assert out.loc[2, "Age"] == 2
    assert out.loc[2, "IsChild"] == "yes"


def test_family_and_fare_features():
    out = TitanicFeatures().fit(sample()).transform(sample())
    assert out.loc[2, "FamilySize"] == 5
    assert out.loc[2, "FamilyType"] == "Large"
    assert out.loc[1, "Deck"] == "C"
    assert out.loc[0, "Deck"] == "U"
    assert abs(out.loc[1, "FarePerPerson"] - 71.28 / 2) < 1e-6


def test_unseen_ticket_and_missing_fare_at_predict_time():
    fe = TitanicFeatures().fit(sample())
    new = sample().iloc[[0]].assign(Ticket="NEW123", Fare=None)
    out = fe.transform(new)
    assert out.loc[0, "TicketGroupSize"] == 2  # falls back to family size
    assert not out.isna().any().any()
