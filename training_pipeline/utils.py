import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path):
    df = pd.read_parquet(path)
    if "user_id" in df.columns:
        df = df.drop(columns=["user_id"])
    X = df.drop(columns=["default"])
    y = df["default"]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def calculate_income_to_loan_ratio(income, loan):
    if loan == 0:
        return 0
    return round(income / loan, 2)
