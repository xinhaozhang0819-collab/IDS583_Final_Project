def load_data(path):
    import pandas as pd

    return pd.read_csv(path)


def create_target(df):
    # turn loan_status into default (1/0)
    df["default"] = df["loan_status"].apply(lambda x: 1 if x == "Charged Off" else 0)
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
