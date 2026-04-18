import pandas as pd
import numpy as np


def load_data(path):
    return pd.read_csv(path)


def create_target(df):
    # Keep Fully Paid and Charged Off only
    df = df[df["loan_status"].isin(["Fully Paid", "Charged Off"])]

    df["default"] = (df["loan_status"] == "Charged Off").astype(int)
    return df


def select_features(df):
    cols = [
        "loan_amnt",
        "annual_inc",
        "fico_range_low",
        "dti",
        "emp_length",
        "home_ownership",
        "term",
        "purpose",
    ]
    return df[cols + ["default"]]


def preprocess_features(df):

    # clean emp_length
    def clean_emp_length(x):
        if pd.isnull(x):
            return np.nan
        x = str(x)
        if "10+" in x:
            return 10
        elif "< 1" in x:
            return 0
        else:
            return int(x.split()[0])

    df["emp_length"] = df["emp_length"].apply(clean_emp_length)

    # clean term
    df["term"] = df["term"].str.extract("(\d+)").astype(float)

    # missing values filling
    df["dti"] = df["dti"].fillna(df["dti"].median())
    df["emp_length"] = df["emp_length"].fillna(df["emp_length"].median())

    # categorical encoding
    df = pd.get_dummies(df, columns=["home_ownership", "purpose"], drop_first=True)

    return df
