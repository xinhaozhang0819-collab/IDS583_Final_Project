import re

import numpy as np
import pandas as pd


TARGET_COLUMN = "default"
ID_COLUMN = "sample_id"
STATUS_COLUMN = "loan_status"
ISSUE_RAW_COLUMN = "issue_d"
ISSUE_DATE_COLUMN = "issue_date"
METADATA_COLUMNS = [ID_COLUMN, ISSUE_DATE_COLUMN]

BASE_RAW_FEATURE_COLUMNS = [
    "loan_amnt",
    "annual_inc",
    "fico_range_low",
    "dti",
    "emp_length",
    "home_ownership",
    "term",
    "purpose",
]
ADDITIONAL_NUMERIC_RAW_FEATURE_COLUMNS = [
    "int_rate",
    "installment",
    "delinq_2yrs",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "mort_acc",
    "pub_rec_bankruptcies",
]
ADDITIONAL_CATEGORICAL_RAW_FEATURE_COLUMNS = [
    "verification_status",
    "grade",
    "initial_list_status",
    "application_type",
]
RAW_FEATURE_COLUMNS = (
    BASE_RAW_FEATURE_COLUMNS
    + ADDITIONAL_NUMERIC_RAW_FEATURE_COLUMNS
    + ADDITIONAL_CATEGORICAL_RAW_FEATURE_COLUMNS
)

BASE_FEATURE_COLUMNS = [
    "loan_amnt",
    "annual_inc",
    "fico_range_low",
    "dti",
    "emp_length",
    "term_months",
]
ADDITIONAL_NUMERIC_FEATURE_COLUMNS = [
    "int_rate",
    "installment",
    "delinq_2yrs",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "mort_acc",
    "pub_rec_bankruptcies",
]
BASE_ENGINEERED_FEATURE_COLUMNS = [
    "missing_emp_length_flag",
    "log_annual_inc",
    "loan_to_income",
    "high_dti_flag",
    "long_term_flag",
]
EXTENDED_ENGINEERED_FEATURE_COLUMNS = [
    "installment_to_income",
    "revol_bal_to_income",
    "high_revol_util_flag",
    "recent_inquiry_flag",
    "prior_delinquency_flag",
    "bankruptcy_flag",
]
ENGINEERED_FEATURE_COLUMNS = (
    BASE_ENGINEERED_FEATURE_COLUMNS + EXTENDED_ENGINEERED_FEATURE_COLUMNS
)

CATEGORICAL_PREFIX_GROUPS = {
    "home_ownership": "home_ownership_",
    "purpose": "purpose_",
    "fico_bucket": "fico_bucket_",
    "verification_status": "verification_status_",
    "grade": "grade_",
    "initial_list_status": "initial_list_status_",
    "application_type": "application_type_",
}
LEGACY_BASE_CATEGORIES = {
    "home_ownership": "ANY",
    "purpose": "car",
    "verification_status": "Not Verified",
    "grade": "A",
    "initial_list_status": "f",
    "application_type": "Individual",
}

DTI_UPPER_BOUND = 80.0
RATE_UPPER_BOUND = 40.0
REVOL_UTIL_UPPER_BOUND = 150.0
HIGH_DTI_THRESHOLD = 30.0
HIGH_REVOL_UTIL_THRESHOLD = 80.0
RECENT_INQUIRY_THRESHOLD = 3.0
FICO_BUCKET_BINS = [0, 660, 700, 740, np.inf]
FICO_BUCKET_LABELS = ["subprime", "fair", "good", "very_good"]

CLEANING_METHODS = [
    "Filter the target to Fully Paid and Charged Off loans only.",
    "Parse issue_d into issue_date and keep it strictly as temporal metadata.",
    "Parse emp_length into numeric years and keep a missing-value flag.",
    "Extract term_months from the original term string.",
    "Convert interest-rate and revolving-utilization percent strings into numeric values.",
    "Convert invalid DTI values below 0 or above 80 to missing before imputation.",
    "Convert invalid int_rate below 0 or above 40 to missing before imputation.",
    "Convert invalid revol_util below 0 or above 150 to missing before imputation.",
    "Impute all modeled numeric columns with the median inside the processed dataset.",
    "One-hot encode home_ownership, purpose, fico_bucket, verification_status, grade, initial_list_status, and application_type with drop_first=True.",
]
FEATURE_GROUPS = {
    "core_numeric_features": BASE_FEATURE_COLUMNS,
    "additional_numeric_features": ADDITIONAL_NUMERIC_FEATURE_COLUMNS,
    "base_engineered_features": BASE_ENGINEERED_FEATURE_COLUMNS,
    "extended_engineered_features": EXTENDED_ENGINEERED_FEATURE_COLUMNS,
    "categorical_dummy_groups": [
        "home_ownership_*",
        "purpose_*",
        "fico_bucket_*",
        "verification_status_*",
        "grade_*",
        "initial_list_status_*",
        "application_type_*",
    ],
}
FEATURE_FLAG_CANDIDATES = {
    "use_additional_numeric_features": "Additional origination-time numeric credit features",
    "use_extended_engineered_features": "Extended engineered ratios and delinquency flags",
    "use_verification_status_dummies": "Verification status dummies",
    "use_grade_dummies": "Grade dummies",
    "use_initial_list_status_dummies": "Initial list status dummies",
    "use_application_type_dummies": "Application type dummies",
}


def load_raw_data(path, **read_csv_kwargs):
    read_options = {"low_memory": False}
    read_options.update(read_csv_kwargs)
    return pd.read_csv(path, **read_options)


def load_processed_data(path, auto_upgrade=True):
    df = pd.read_csv(path)
    if ISSUE_DATE_COLUMN in df.columns:
        df[ISSUE_DATE_COLUMN] = pd.to_datetime(df[ISSUE_DATE_COLUMN], errors="coerce")
    if auto_upgrade:
        return ensure_model_ready_dataset(df)
    return df


def ensure_sample_id(df, id_column=ID_COLUMN):
    if id_column in df.columns:
        return df.copy()

    enriched = df.copy()
    enriched[id_column] = np.arange(len(enriched))
    return enriched


def parse_issue_date_column(
    df,
    raw_column=ISSUE_RAW_COLUMN,
    parsed_column=ISSUE_DATE_COLUMN,
):
    enriched = ensure_sample_id(df)
    parsed = enriched.copy()

    if parsed_column in parsed.columns:
        parsed[parsed_column] = pd.to_datetime(parsed[parsed_column], errors="coerce")
        return parsed

    if raw_column not in parsed.columns:
        raise KeyError(f"Missing both {parsed_column} and {raw_column}.")

    parsed[parsed_column] = pd.to_datetime(
        parsed[raw_column],
        format="%b-%Y",
        errors="coerce",
    )
    return parsed


def create_target(df, status_column=STATUS_COLUMN, target_column=TARGET_COLUMN):
    enriched = ensure_sample_id(df)
    filtered = enriched[enriched[status_column].isin(["Fully Paid", "Charged Off"])].copy()
    filtered[target_column] = (filtered[status_column] == "Charged Off").astype(int)
    return filtered


def select_modeling_columns(df, include_target=True):
    required_columns = METADATA_COLUMNS + RAW_FEATURE_COLUMNS
    if include_target:
        required_columns = required_columns + [TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing required modeling columns: {missing_columns}")

    return df[required_columns].copy()


def parse_emp_length_value(value):
    if pd.isna(value):
        return np.nan

    text = str(value).strip().lower()
    if not text or text in {"nan", "n/a"}:
        return np.nan
    if "10+" in text:
        return 10.0
    if "< 1" in text:
        return 0.0

    match = re.search(r"(\d+)", text)
    return float(match.group(1)) if match else np.nan


def clean_base_features(
    df,
    dti_upper_bound=DTI_UPPER_BOUND,
    rate_upper_bound=RATE_UPPER_BOUND,
    revol_util_upper_bound=REVOL_UTIL_UPPER_BOUND,
):
    cleaned = df.copy()
    cleaned[ISSUE_DATE_COLUMN] = pd.to_datetime(cleaned[ISSUE_DATE_COLUMN], errors="coerce")

    numeric_columns = [
        "loan_amnt",
        "annual_inc",
        "fico_range_low",
        "dti",
        "installment",
        "delinq_2yrs",
        "inq_last_6mths",
        "open_acc",
        "pub_rec",
        "revol_bal",
        "total_acc",
        "mort_acc",
        "pub_rec_bankruptcies",
    ]
    for column in numeric_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned["int_rate"] = _parse_percent_series(cleaned["int_rate"])
    cleaned["revol_util"] = _parse_percent_series(cleaned["revol_util"])
    cleaned["emp_length"] = cleaned["emp_length"].apply(parse_emp_length_value)
    cleaned["missing_emp_length_flag"] = cleaned["emp_length"].isna().astype(int)
    cleaned["term_months"] = pd.to_numeric(
        cleaned["term"].astype(str).str.extract(r"(\d+)")[0],
        errors="coerce",
    )

    cleaned.loc[(cleaned["dti"] < 0) | (cleaned["dti"] > dti_upper_bound), "dti"] = np.nan
    cleaned.loc[
        (cleaned["int_rate"] < 0) | (cleaned["int_rate"] > rate_upper_bound),
        "int_rate",
    ] = np.nan
    cleaned.loc[
        (cleaned["revol_util"] < 0) | (cleaned["revol_util"] > revol_util_upper_bound),
        "revol_util",
    ] = np.nan

    non_negative_columns = [
        "annual_inc",
        "installment",
        "delinq_2yrs",
        "inq_last_6mths",
        "open_acc",
        "pub_rec",
        "revol_bal",
        "total_acc",
        "mort_acc",
        "pub_rec_bankruptcies",
        "term_months",
    ]
    for column in non_negative_columns:
        cleaned.loc[cleaned[column] < 0, column] = np.nan

    modeled_numeric_columns = (
        BASE_FEATURE_COLUMNS + ADDITIONAL_NUMERIC_FEATURE_COLUMNS
    )
    for column in modeled_numeric_columns:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].fillna(cleaned[column].median())

    return cleaned


def add_engineered_features(
    df,
    high_dti_threshold=HIGH_DTI_THRESHOLD,
    high_revol_util_threshold=HIGH_REVOL_UTIL_THRESHOLD,
    recent_inquiry_threshold=RECENT_INQUIRY_THRESHOLD,
    fico_bins=FICO_BUCKET_BINS,
    fico_labels=FICO_BUCKET_LABELS,
):
    engineered = df.copy()

    safe_income = engineered["annual_inc"].clip(lower=0)
    income_denominator = safe_income.replace(0, np.nan)

    engineered["log_annual_inc"] = np.log1p(safe_income)
    engineered["loan_to_income"] = engineered["loan_amnt"] / income_denominator
    engineered["loan_to_income"] = _clean_ratio(engineered["loan_to_income"])
    engineered["high_dti_flag"] = (engineered["dti"] >= high_dti_threshold).astype(int)
    engineered["long_term_flag"] = (engineered["term_months"] >= 60).astype(int)

    engineered["installment_to_income"] = engineered["installment"] / income_denominator
    engineered["installment_to_income"] = _clean_ratio(engineered["installment_to_income"])
    engineered["revol_bal_to_income"] = engineered["revol_bal"] / income_denominator
    engineered["revol_bal_to_income"] = _clean_ratio(engineered["revol_bal_to_income"])
    engineered["high_revol_util_flag"] = (
        engineered["revol_util"] >= high_revol_util_threshold
    ).astype(int)
    engineered["recent_inquiry_flag"] = (
        engineered["inq_last_6mths"] >= recent_inquiry_threshold
    ).astype(int)
    engineered["prior_delinquency_flag"] = (engineered["delinq_2yrs"] > 0).astype(int)
    engineered["bankruptcy_flag"] = (engineered["pub_rec_bankruptcies"] > 0).astype(int)

    fico_bucket = pd.cut(
        engineered["fico_range_low"],
        bins=fico_bins,
        labels=fico_labels,
        right=False,
    )
    engineered["fico_bucket"] = fico_bucket.astype("object").fillna("unknown")

    return engineered


def encode_categorical_features(df):
    encoded = df.copy()
    encoded = encoded.drop(columns=["term"], errors="ignore")

    categorical_columns = [
        column
        for column in [
            "home_ownership",
            "purpose",
            "fico_bucket",
            "verification_status",
            "grade",
            "initial_list_status",
            "application_type",
        ]
        if column in encoded.columns
    ]

    encoded = pd.get_dummies(
        encoded,
        columns=categorical_columns,
        drop_first=True,
        dtype=int,
    )

    return reorder_modeling_columns(encoded)


def build_modeling_dataset(df):
    working = ensure_sample_id(df)
    working = parse_issue_date_column(working)
    if TARGET_COLUMN not in working.columns:
        working = create_target(working)

    working = select_modeling_columns(working)
    working = clean_base_features(working)
    working = add_engineered_features(working)
    working = encode_categorical_features(working)
    return reorder_modeling_columns(working)


def build_feature_dataset_from_raw(df):
    working = ensure_sample_id(df)
    working = parse_issue_date_column(working)
    working = select_modeling_columns(working, include_target=TARGET_COLUMN in working.columns)
    working = clean_base_features(working)
    working = add_engineered_features(working)
    working = encode_categorical_features(working)
    return reorder_modeling_columns(working)


def upgrade_legacy_processed_data(df):
    upgraded = ensure_sample_id(df)

    if ISSUE_DATE_COLUMN in upgraded.columns:
        upgraded[ISSUE_DATE_COLUMN] = pd.to_datetime(upgraded[ISSUE_DATE_COLUMN], errors="coerce")
    elif ISSUE_RAW_COLUMN in upgraded.columns:
        upgraded[ISSUE_DATE_COLUMN] = pd.to_datetime(
            upgraded[ISSUE_RAW_COLUMN],
            format="%b-%Y",
            errors="coerce",
        )
    else:
        upgraded[ISSUE_DATE_COLUMN] = pd.NaT

    if "term_months" not in upgraded.columns and "term" in upgraded.columns:
        upgraded["term_months"] = pd.to_numeric(
            upgraded["term"].astype(str).str.extract(r"(\d+)")[0],
            errors="coerce",
        )

    if "missing_emp_length_flag" not in upgraded.columns:
        upgraded["missing_emp_length_flag"] = 0

    if "log_annual_inc" not in upgraded.columns:
        upgraded["log_annual_inc"] = np.log1p(upgraded["annual_inc"].clip(lower=0))

    if "loan_to_income" not in upgraded.columns:
        income_denominator = upgraded["annual_inc"].replace(0, np.nan)
        upgraded["loan_to_income"] = upgraded["loan_amnt"] / income_denominator
        upgraded["loan_to_income"] = _clean_ratio(upgraded["loan_to_income"])

    if "high_dti_flag" not in upgraded.columns:
        upgraded["high_dti_flag"] = (upgraded["dti"] >= HIGH_DTI_THRESHOLD).astype(int)

    if "long_term_flag" not in upgraded.columns:
        upgraded["long_term_flag"] = (upgraded["term_months"] >= 60).astype(int)

    if "installment_to_income" not in upgraded.columns and {
        "installment",
        "annual_inc",
    }.issubset(upgraded.columns):
        income_denominator = upgraded["annual_inc"].replace(0, np.nan)
        upgraded["installment_to_income"] = upgraded["installment"] / income_denominator
        upgraded["installment_to_income"] = _clean_ratio(upgraded["installment_to_income"])

    if "revol_bal_to_income" not in upgraded.columns and {
        "revol_bal",
        "annual_inc",
    }.issubset(upgraded.columns):
        income_denominator = upgraded["annual_inc"].replace(0, np.nan)
        upgraded["revol_bal_to_income"] = upgraded["revol_bal"] / income_denominator
        upgraded["revol_bal_to_income"] = _clean_ratio(upgraded["revol_bal_to_income"])

    if "high_revol_util_flag" not in upgraded.columns and "revol_util" in upgraded.columns:
        upgraded["high_revol_util_flag"] = (
            upgraded["revol_util"] >= HIGH_REVOL_UTIL_THRESHOLD
        ).astype(int)

    if "recent_inquiry_flag" not in upgraded.columns and "inq_last_6mths" in upgraded.columns:
        upgraded["recent_inquiry_flag"] = (
            upgraded["inq_last_6mths"] >= RECENT_INQUIRY_THRESHOLD
        ).astype(int)

    if "prior_delinquency_flag" not in upgraded.columns and "delinq_2yrs" in upgraded.columns:
        upgraded["prior_delinquency_flag"] = (upgraded["delinq_2yrs"] > 0).astype(int)

    if "bankruptcy_flag" not in upgraded.columns and "pub_rec_bankruptcies" in upgraded.columns:
        upgraded["bankruptcy_flag"] = (upgraded["pub_rec_bankruptcies"] > 0).astype(int)

    if not any(column.startswith("fico_bucket_") for column in upgraded.columns):
        fico_bucket = pd.cut(
            upgraded["fico_range_low"],
            bins=FICO_BUCKET_BINS,
            labels=FICO_BUCKET_LABELS,
            right=False,
        )
        fico_bucket = fico_bucket.astype("object").fillna("unknown")
        fico_dummies = pd.get_dummies(
            fico_bucket,
            prefix="fico_bucket",
            drop_first=True,
            dtype=int,
        )
        upgraded = pd.concat([upgraded, fico_dummies], axis=1)

    return reorder_modeling_columns(upgraded)


def ensure_model_ready_dataset(df):
    working = ensure_sample_id(df)
    has_raw_categories = {"home_ownership", "purpose"}.issubset(working.columns)
    has_raw_term = "term" in working.columns

    if has_raw_categories and has_raw_term:
        working = parse_issue_date_column(working)
        if TARGET_COLUMN not in working.columns:
            if STATUS_COLUMN not in working.columns:
                raise KeyError("The dataset is missing both default and loan_status columns.")
            working = create_target(working)
        if set(RAW_FEATURE_COLUMNS).issubset(working.columns):
            working = build_feature_dataset_from_raw(working)
            return reorder_modeling_columns(working)

    return upgrade_legacy_processed_data(working)


def reconstruct_raw_like_dataset(df):
    reconstructed = ensure_model_ready_dataset(df)
    raw_like = pd.DataFrame()

    raw_like[ID_COLUMN] = reconstructed[ID_COLUMN]
    raw_like[ISSUE_RAW_COLUMN] = reconstructed[ISSUE_DATE_COLUMN].dt.strftime("%b-%Y")
    raw_like[ISSUE_DATE_COLUMN] = reconstructed[ISSUE_DATE_COLUMN]
    raw_like["loan_amnt"] = reconstructed["loan_amnt"]
    raw_like["annual_inc"] = reconstructed["annual_inc"]
    raw_like["fico_range_low"] = reconstructed["fico_range_low"]
    raw_like["dti"] = reconstructed["dti"]
    raw_like["emp_length"] = reconstructed["emp_length"].apply(_format_emp_length_string)
    raw_like["term"] = reconstructed["term_months"].round().astype(int).astype(str) + " months"

    for column in ADDITIONAL_NUMERIC_FEATURE_COLUMNS:
        if column in reconstructed.columns:
            raw_like[column] = reconstructed[column]

    raw_like["home_ownership"] = _decode_categorical_group(
        reconstructed,
        CATEGORICAL_PREFIX_GROUPS["home_ownership"],
        LEGACY_BASE_CATEGORIES["home_ownership"],
    )
    raw_like["purpose"] = _decode_categorical_group(
        reconstructed,
        CATEGORICAL_PREFIX_GROUPS["purpose"],
        LEGACY_BASE_CATEGORIES["purpose"],
    )
    raw_like["verification_status"] = _decode_categorical_group(
        reconstructed,
        CATEGORICAL_PREFIX_GROUPS["verification_status"],
        LEGACY_BASE_CATEGORIES["verification_status"],
    )
    raw_like["grade"] = _decode_categorical_group(
        reconstructed,
        CATEGORICAL_PREFIX_GROUPS["grade"],
        LEGACY_BASE_CATEGORIES["grade"],
    )
    raw_like["initial_list_status"] = _decode_categorical_group(
        reconstructed,
        CATEGORICAL_PREFIX_GROUPS["initial_list_status"],
        LEGACY_BASE_CATEGORIES["initial_list_status"],
    )
    raw_like["application_type"] = _decode_categorical_group(
        reconstructed,
        CATEGORICAL_PREFIX_GROUPS["application_type"],
        LEGACY_BASE_CATEGORIES["application_type"],
    )
    raw_like[STATUS_COLUMN] = reconstructed[TARGET_COLUMN].map(
        {0: "Fully Paid", 1: "Charged Off"}
    )
    raw_like[TARGET_COLUMN] = reconstructed[TARGET_COLUMN]

    return raw_like


def select_feature_columns(df, feature_flags=None):
    feature_flags = feature_flags or {}

    selected_columns = [column for column in BASE_FEATURE_COLUMNS if column in df.columns]

    if feature_flags.get("use_additional_numeric_features", True):
        selected_columns.extend(
            [column for column in ADDITIONAL_NUMERIC_FEATURE_COLUMNS if column in df.columns]
        )

    if feature_flags.get("use_engineered_features", True):
        selected_columns.extend(
            [column for column in BASE_ENGINEERED_FEATURE_COLUMNS if column in df.columns]
        )

    if feature_flags.get("use_extended_engineered_features", True):
        selected_columns.extend(
            [column for column in EXTENDED_ENGINEERED_FEATURE_COLUMNS if column in df.columns]
        )

    if feature_flags.get("use_home_ownership_dummies", True):
        selected_columns.extend(_prefixed_columns(df, CATEGORICAL_PREFIX_GROUPS["home_ownership"]))

    if feature_flags.get("use_purpose_dummies", True):
        selected_columns.extend(_prefixed_columns(df, CATEGORICAL_PREFIX_GROUPS["purpose"]))

    if feature_flags.get("use_fico_bucket_dummies", True):
        selected_columns.extend(_prefixed_columns(df, CATEGORICAL_PREFIX_GROUPS["fico_bucket"]))

    if feature_flags.get("use_verification_status_dummies", False):
        selected_columns.extend(
            _prefixed_columns(df, CATEGORICAL_PREFIX_GROUPS["verification_status"])
        )

    if feature_flags.get("use_grade_dummies", False):
        selected_columns.extend(_prefixed_columns(df, CATEGORICAL_PREFIX_GROUPS["grade"]))

    if feature_flags.get("use_initial_list_status_dummies", False):
        selected_columns.extend(
            _prefixed_columns(df, CATEGORICAL_PREFIX_GROUPS["initial_list_status"])
        )

    if feature_flags.get("use_application_type_dummies", False):
        selected_columns.extend(
            _prefixed_columns(df, CATEGORICAL_PREFIX_GROUPS["application_type"])
        )

    return list(dict.fromkeys(selected_columns))


def reorder_modeling_columns(df):
    ordered_columns = []
    ordered_columns.extend([column for column in METADATA_COLUMNS if column in df.columns])
    ordered_columns.extend([column for column in BASE_FEATURE_COLUMNS if column in df.columns])
    ordered_columns.extend(
        [column for column in ADDITIONAL_NUMERIC_FEATURE_COLUMNS if column in df.columns]
    )
    ordered_columns.extend(
        [column for column in BASE_ENGINEERED_FEATURE_COLUMNS if column in df.columns]
    )
    ordered_columns.extend(
        [column for column in EXTENDED_ENGINEERED_FEATURE_COLUMNS if column in df.columns]
    )

    for prefix in CATEGORICAL_PREFIX_GROUPS.values():
        ordered_columns.extend(_prefixed_columns(df, prefix))

    remaining_columns = [
        column
        for column in df.columns
        if column not in ordered_columns and column != TARGET_COLUMN
    ]
    ordered_columns.extend(remaining_columns)

    if TARGET_COLUMN in df.columns:
        ordered_columns.append(TARGET_COLUMN)

    unique_columns = []
    seen = set()
    for column in ordered_columns:
        if column in df.columns and column not in seen:
            unique_columns.append(column)
            seen.add(column)

    return df[unique_columns].copy()


def _prefixed_columns(df, prefix):
    return sorted([column for column in df.columns if column.startswith(prefix)])


def _parse_percent_series(series):
    text_series = series.astype(str).str.replace("%", "", regex=False).str.strip()
    return pd.to_numeric(text_series, errors="coerce")


def _clean_ratio(series):
    cleaned = series.replace([np.inf, -np.inf], np.nan)
    return cleaned.fillna(cleaned.median())


def _decode_categorical_group(df, prefix, base_label):
    group_columns = _prefixed_columns(df, prefix)
    if not group_columns:
        return pd.Series(base_label, index=df.index)

    def _decode_row(row):
        active_columns = [column for column in group_columns if row[column] == 1]
        if not active_columns:
            return base_label
        return active_columns[0].replace(prefix, "", 1)

    return df[group_columns].apply(_decode_row, axis=1)


def _format_emp_length_string(value):
    if pd.isna(value):
        return np.nan
    if value <= 0:
        return "< 1 year"
    if value >= 10:
        return "10+ years"
    rounded = int(round(value))
    suffix = "year" if rounded == 1 else "years"
    return f"{rounded} {suffix}"
