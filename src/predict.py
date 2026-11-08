"""Predict survival for a single passenger from the command line.

Example:
  python src/predict.py --name "Smith, Mrs. Anna" --sex female --age 29 \
      --pclass 2 --sibsp 1 --parch 1 --fare 26 --embarked S
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description="Would this passenger have survived the Titanic?")
    p.add_argument("--name", required=True, help='Format "Surname, Title. First names"')
    p.add_argument("--sex", choices=["male", "female"], required=True)
    p.add_argument("--age", type=float, default=None)
    p.add_argument("--pclass", type=int, choices=[1, 2, 3], required=True)
    p.add_argument("--sibsp", type=int, default=0, help="siblings/spouses aboard")
    p.add_argument("--parch", type=int, default=0, help="parents/children aboard")
    p.add_argument("--fare", type=float, default=None)
    p.add_argument("--embarked", choices=["S", "C", "Q"], default="S")
    p.add_argument("--cabin", default=None, help='e.g. "C85"')
    p.add_argument("--ticket", default="UNKNOWN")
    a = p.parse_args()

    model = joblib.load(ROOT / "models" / "titanic_model.joblib")
    passenger = pd.DataFrame([{
        "PassengerId": 0, "Pclass": a.pclass, "Name": a.name, "Sex": a.sex, "Age": a.age,
        "SibSp": a.sibsp, "Parch": a.parch, "Ticket": a.ticket, "Fare": a.fare,
        "Cabin": a.cabin, "Embarked": a.embarked,
    }])
    prob = model.predict_proba(passenger)[0, 1]
    print(f"Survival probability: {prob:.1%} -> {'SURVIVED' if prob >= 0.5 else 'DID NOT SURVIVE'}")


if __name__ == "__main__":
    main()
