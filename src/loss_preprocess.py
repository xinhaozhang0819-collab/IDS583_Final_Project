import numpy as np
import pandas as pd

from preprocess import (
    FICO_BUCKET_BINS,
    FICO_BUCKET_LABELS,
    ID_COLUMN,
    ISSUE_DATE_COLUMN,
    ISSUE_RAW_COLUMN,
    RAW_FEATURE_COLUMNS,
    RATE_UPPER_BOUND,
    STATUS_COLUMN,
    TARGET_COLUMN,
    ensure_sample_id,
    load_raw_data,
)
from split import DEFAULT_TEMPORAL_SPLIT, assign_temporal_split_labels


LAST_PAYMENT_RAW_COLUMN = "last_pymnt_d"
LAST_PAYMENT_DATE_COLUMN = "last_payment_date"
SPLIT_LABEL_COLUMN = "split_label"

LOSS_STATUS_VALUES = [
    "Fully Paid",
    "Charged Off",
    "Current",
    "In Grace Period",
    "Late (16-30 days)",
    "Late (31-120 days)",
    "Default",
]
RESOLVED_LOAN_STATUSES = ["Fully Paid", "Charged Off"]
ACTIVE_LOAN_STATUSES = [
    "Current",
    "In Grace Period",
    "Late (16-30 days)",
    "Late (31-120 days)",
    "Default",
]
CHARGED_OFF_STATUS = "Charged Off"

LOSS_RAW_COLUMNS = list(
    dict.fromkeys(
        [
            STATUS_COLUMN,
            ISSUE_RAW_COLUMN,
            LAST_PAYMENT_RAW_COLUMN,
            *RAW_FEATURE_COLUMNS,
            "funded_amnt",
            "funded_amnt_inv",
            "total_pymnt",
            "total_pymnt_inv",
            "total_rec_prncp",
            "total_rec_int",
            "recoveries",
            "collection_recovery_fee",
            "last_pymnt_amnt",
            "out_prncp",
            "out_prncp_inv",
        ]
    )
)

LOSS_CLEANING_METHODS = [
    "Load origination and performance fields required for LGD, EAD, EL, and reserve analytics.",
    "Parse issue_d and last_pymnt_d into issue_date and last_payment_date.",
    "Keep resolved, current, grace-period, late, and default statuses needed for event and censoring logic.",
    "Convert funded amount, balances, payments, recoveries, installment, income, and FICO fields to numeric.",
    "Parse term into term_months and interest rate into a numeric annual percentage.",
    "Cap months_on_book at term_months for amortization-aware reserve analytics.",
    "Map purpose into management buckets and annual income into reporting bands.",
    "Use temporal split labels based on origination date so future vintages never enter training.",
]

PURPOSE_GROUP_MAP = {
    "debt_consolidation": "debt_consolidation",
    "credit_card": "credit_card",
    "home_improvement": "home_improvement",
    "major_purchase": "major_purchase",
}
PURPOSE_GROUP_DEFAULT = "other"
ANNUAL_INCOME_BINS = [0, 50000, 100000, 150000, np.inf]
ANNUAL_INCOME_LABELS = ["<50k", "50-100k", "100-150k", "150k+"]
LGD_SEGMENT_KEYS = ["grade", "term_months", "issue_year"]
MANAGERIAL_SEGMENT_COLUMNS = [
    "grade",
    "fico_bucket",
    "term_months",
    "issue_year",
    "issue_year_quarter",
    "purpose_group",
    "annual_income_band",
]


def load_loss_raw_dataset(path, **read_csv_kwargs):
    read_options = {"usecols": LOSS_RAW_COLUMNS}
    read_options.update(read_csv_kwargs)
    return load_raw_data(path, **read_options)


def prepare_loss_workflow_dataset(
    df,
    train_end=DEFAULT_TEMPORAL_SPLIT["train_end"],
    valid_end=DEFAULT_TEMPORAL_SPLIT["valid_end"],
):
    prepared = ensure_sample_id(df)
    prepared = prepared[prepared[STATUS_COLUMN].isin(LOSS_STATUS_VALUES)].copy()

    prepared[ISSUE_DATE_COLUMN] = pd.to_datetime(
        prepared[ISSUE_RAW_COLUMN],
        format="%b-%Y",
        errors="coerce",
    )
    prepared[LAST_PAYMENT_DATE_COLUMN] = pd.to_datetime(
        prepared[LAST_PAYMENT_RAW_COLUMN],
        format="%b-%Y",
        errors="coerce",
    )

    numeric_columns = [
        "funded_amnt",
        "funded_amnt_inv",
        "loan_amnt",
        "installment",
        "out_prncp",
        "out_prncp_inv",
        "total_pymnt",
        "total_pymnt_inv",
        "total_rec_prncp",
        "total_rec_int",
        "recoveries",
        "collection_recovery_fee",
        "last_pymnt_amnt",
        "annual_inc",
        "fico_range_low",
    ]
    for column in numeric_columns:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    prepared["int_rate"] = _parse_percent_series(prepared["int_rate"])
    prepared.loc[
        (prepared["int_rate"] < 0) | (prepared["int_rate"] > RATE_UPPER_BOUND),
        "int_rate",
    ] = np.nan

    prepared["term_months"] = pd.to_numeric(
        prepared["term"].astype(str).str.extract(r"(\d+)")[0],
        errors="coerce",
    )
    prepared["term_months"] = prepared["term_months"].fillna(0).clip(lower=0)

    prepared["months_on_book_raw"] = month_diff(
        prepared[ISSUE_DATE_COLUMN],
        prepared[LAST_PAYMENT_DATE_COLUMN],
    )
    prepared["months_on_book_raw"] = prepared["months_on_book_raw"].clip(lower=0)
    prepared["months_on_book"] = np.minimum(
        prepared["months_on_book_raw"],
        prepared["term_months"],
    )
    prepared["remaining_term"] = (
        prepared["term_months"] - prepared["months_on_book"]
    ).clip(lower=0)

    prepared["issue_year"] = prepared[ISSUE_DATE_COLUMN].dt.year.astype("Int64")
    prepared["issue_year_quarter"] = _year_quarter_label(prepared[ISSUE_DATE_COLUMN])
    prepared["fico_bucket"] = pd.cut(
        prepared["fico_range_low"],
        bins=FICO_BUCKET_BINS,
        labels=FICO_BUCKET_LABELS,
        right=False,
    ).astype("object").fillna("unknown")
    prepared["purpose_group"] = (
        prepared["purpose"].map(PURPOSE_GROUP_MAP).fillna(PURPOSE_GROUP_DEFAULT)
    )
    prepared["annual_income_band"] = pd.cut(
        prepared["annual_inc"].clip(lower=0),
        bins=ANNUAL_INCOME_BINS,
        labels=ANNUAL_INCOME_LABELS,
        right=False,
        include_lowest=True,
    ).astype("object").fillna(ANNUAL_INCOME_LABELS[0])

    prepared["charged_off_flag"] = (prepared[STATUS_COLUMN] == CHARGED_OFF_STATUS).astype(int)
    prepared["resolved_flag"] = prepared[STATUS_COLUMN].isin(RESOLVED_LOAN_STATUSES).astype(int)
    prepared["active_flag"] = prepared[STATUS_COLUMN].isin(ACTIVE_LOAN_STATUSES).astype(int)
    prepared["actual_default_12m"] = (
        (prepared["charged_off_flag"] == 1) & (prepared["months_on_book"] <= 12)
    ).astype(int)

    prepared["scheduled_balance_proxy"] = scheduled_balance_proxy(
        funded_amount=prepared["funded_amnt"],
        installment=prepared["installment"],
        annual_rate=prepared["int_rate"],
        months_elapsed=prepared["months_on_book"],
    )
    prepared["ead_current"] = np.where(
        prepared["out_prncp"].fillna(0) > 0,
        prepared["out_prncp"],
        prepared["scheduled_balance_proxy"],
    )
    prepared["ead_current"] = pd.to_numeric(prepared["ead_current"], errors="coerce").fillna(0)
    prepared["ead_current"] = prepared["ead_current"].clip(lower=0)

    prepared["net_recoveries"] = (
        prepared["recoveries"].fillna(0) - prepared["collection_recovery_fee"].fillna(0)
    ).clip(lower=0)
    prepared["actual_net_loss"] = (
        prepared["funded_amnt"].fillna(0)
        - prepared["total_rec_prncp"].fillna(0)
        - prepared["net_recoveries"]
    ).clip(lower=0)
    prepared["actual_loss_rate"] = np.where(
        prepared["funded_amnt"].fillna(0) > 0,
        prepared["actual_net_loss"] / prepared["funded_amnt"],
        0.0,
    )

    prepared = assign_temporal_split_labels(
        prepared,
        date_col=ISSUE_DATE_COLUMN,
        train_end=train_end,
        valid_end=valid_end,
        output_col=SPLIT_LABEL_COLUMN,
    )

    ordered_columns = [
        ID_COLUMN,
        ISSUE_DATE_COLUMN,
        LAST_PAYMENT_DATE_COLUMN,
        SPLIT_LABEL_COLUMN,
        STATUS_COLUMN,
        "charged_off_flag",
        "resolved_flag",
        "active_flag",
        "actual_default_12m",
        "grade",
        "purpose",
        "purpose_group",
        "fico_range_low",
        "fico_bucket",
        "annual_inc",
        "annual_income_band",
        "dti",
        "emp_length",
        "loan_amnt",
        "funded_amnt",
        "home_ownership",
        "verification_status",
        "initial_list_status",
        "application_type",
        "term_months",
        "term",
        "installment",
        "int_rate",
        "delinq_2yrs",
        "inq_last_6mths",
        "open_acc",
        "pub_rec",
        "revol_bal",
        "revol_util",
        "total_acc",
        "mort_acc",
        "pub_rec_bankruptcies",
        "months_on_book_raw",
        "months_on_book",
        "remaining_term",
        "out_prncp",
        "scheduled_balance_proxy",
        "ead_current",
        "total_rec_prncp",
        "recoveries",
        "collection_recovery_fee",
        "net_recoveries",
        "actual_net_loss",
        "actual_loss_rate",
    ]
    available_columns = [column for column in ordered_columns if column in prepared.columns]
    remaining_columns = [
        column
        for column in prepared.columns
        if column not in available_columns
    ]
    return prepared[available_columns + remaining_columns].sort_values(
        [ISSUE_DATE_COLUMN, ID_COLUMN]
    ).reset_index(drop=True)


def build_charged_off_loss_proxy(df):
    charged = df[df[STATUS_COLUMN] == CHARGED_OFF_STATUS].copy()
    charged["ead_proxy"] = (
        charged["funded_amnt"].fillna(0) - charged["total_rec_prncp"].fillna(0)
    ).clip(lower=0)
    charged["net_recoveries"] = (
        charged["recoveries"].fillna(0) - charged["collection_recovery_fee"].fillna(0)
    ).clip(lower=0)
    charged = charged[charged["ead_proxy"] > 0].copy()

    charged["recovery_rate_proxy"] = (
        charged["net_recoveries"] / charged["ead_proxy"]
    ).clip(lower=0, upper=1)
    charged["lgd_proxy"] = (1 - charged["recovery_rate_proxy"]).clip(lower=0, upper=1)
    charged["realized_loss_amount"] = charged["ead_proxy"] * charged["lgd_proxy"]
    return charged.reset_index(drop=True)


def fit_expected_lgd_lookup(charged_off_df, fit_splits=("train", "validation"), shrinkage_k=50):
    fitting = charged_off_df[charged_off_df[SPLIT_LABEL_COLUMN].isin(fit_splits)].copy()
    if fitting.empty:
        raise ValueError("No charged-off rows are available to fit the expected LGD lookup.")

    portfolio_mean = float(fitting["lgd_proxy"].mean())

    exact = _summarize_lgd_group(
        fitting,
        group_columns=LGD_SEGMENT_KEYS,
        mean_column_name="exact_lgd_mean",
        count_column_name="exact_count",
        expected_column_name="expected_lgd_exact",
        portfolio_mean=portfolio_mean,
        shrinkage_k=shrinkage_k,
    )
    grade_term = _summarize_lgd_group(
        fitting,
        group_columns=["grade", "term_months"],
        mean_column_name="grade_term_lgd_mean",
        count_column_name="grade_term_count",
        expected_column_name="expected_lgd_grade_term",
        portfolio_mean=portfolio_mean,
        shrinkage_k=shrinkage_k,
    )
    grade_only = _summarize_lgd_group(
        fitting,
        group_columns=["grade"],
        mean_column_name="grade_lgd_mean",
        count_column_name="grade_count",
        expected_column_name="expected_lgd_grade",
        portfolio_mean=portfolio_mean,
        shrinkage_k=shrinkage_k,
    )

    return {
        "portfolio_mean": portfolio_mean,
        "shrinkage_k": shrinkage_k,
        "exact": exact,
        "grade_term": grade_term,
        "grade_only": grade_only,
        "training_rows": int(len(fitting)),
        "fit_splits": list(fit_splits),
    }


def apply_expected_lgd_lookup(df, lgd_lookup):
    enriched = df.copy()

    exact = lgd_lookup["exact"][
        LGD_SEGMENT_KEYS + ["expected_lgd_exact", "exact_count", "exact_credibility_weight"]
    ]
    grade_term = lgd_lookup["grade_term"][
        ["grade", "term_months", "expected_lgd_grade_term", "grade_term_count"]
    ]
    grade_only = lgd_lookup["grade_only"][
        ["grade", "expected_lgd_grade", "grade_count"]
    ]

    enriched = enriched.merge(exact, on=LGD_SEGMENT_KEYS, how="left")
    enriched = enriched.merge(grade_term, on=["grade", "term_months"], how="left")
    enriched = enriched.merge(grade_only, on=["grade"], how="left")

    portfolio_mean = lgd_lookup["portfolio_mean"]
    enriched["expected_lgd"] = enriched["expected_lgd_exact"]
    enriched["lgd_lookup_source"] = "exact_segment"

    grade_term_mask = enriched["expected_lgd"].isna() & enriched["expected_lgd_grade_term"].notna()
    enriched.loc[grade_term_mask, "expected_lgd"] = enriched.loc[
        grade_term_mask,
        "expected_lgd_grade_term",
    ]
    enriched.loc[grade_term_mask, "lgd_lookup_source"] = "grade_term"

    grade_mask = enriched["expected_lgd"].isna() & enriched["expected_lgd_grade"].notna()
    enriched.loc[grade_mask, "expected_lgd"] = enriched.loc[
        grade_mask,
        "expected_lgd_grade",
    ]
    enriched.loc[grade_mask, "lgd_lookup_source"] = "grade"

    portfolio_mask = enriched["expected_lgd"].isna()
    enriched.loc[portfolio_mask, "expected_lgd"] = portfolio_mean
    enriched.loc[portfolio_mask, "lgd_lookup_source"] = "portfolio"

    enriched["expected_lgd"] = enriched["expected_lgd"].clip(lower=0, upper=1)
    return enriched


def compute_expected_loss(df, pd_column, lgd_column, ead_column, output_column):
    enriched = df.copy()
    enriched[output_column] = (
        enriched[pd_column].fillna(0)
        * enriched[lgd_column].fillna(0)
        * enriched[ead_column].fillna(0)
    )
    return enriched


def build_portfolio_expected_loss_summary(
    df,
    analysis_scope,
    pd_measure,
    pd_column,
    lgd_column,
    ead_column,
    el_column,
):
    total_el = float(df[el_column].sum())
    return pd.DataFrame(
        [
            {
                "analysis_scope": analysis_scope,
                "pd_measure": pd_measure,
                "loan_count": int(len(df)),
                "funded_amount": float(df["funded_amnt"].sum()),
                "avg_pd": float(df[pd_column].mean()),
                "avg_lgd": float(df[lgd_column].mean()),
                "avg_ead": float(df[ead_column].mean()),
                "avg_el": float(df[el_column].mean()),
                "total_el": total_el,
                "portfolio_el_share": 1.0 if total_el > 0 else 0.0,
            }
        ]
    )


def build_managerial_segmentation_summary(
    df,
    analysis_scope,
    pd_measure,
    pd_column,
    lgd_column,
    ead_column,
    el_column,
):
    total_el = float(df[el_column].sum())
    summaries = []

    for segment_column in MANAGERIAL_SEGMENT_COLUMNS:
        if segment_column not in df.columns:
            continue
        grouped = (
            df.groupby(segment_column, dropna=False)
            .agg(
                loan_count=(ID_COLUMN, "size"),
                funded_amount=("funded_amnt", "sum"),
                avg_pd=(pd_column, "mean"),
                avg_lgd=(lgd_column, "mean"),
                avg_ead=(ead_column, "mean"),
                avg_el=(el_column, "mean"),
                total_el=(el_column, "sum"),
            )
            .reset_index()
            .rename(columns={segment_column: "segment_value"})
        )
        grouped["segment_value"] = grouped["segment_value"].astype(str)
        grouped["segment_type"] = segment_column
        grouped["analysis_scope"] = analysis_scope
        grouped["pd_measure"] = pd_measure
        grouped["portfolio_el_share"] = np.where(
            total_el > 0,
            grouped["total_el"] / total_el,
            0.0,
        )
        summaries.append(grouped)

    if not summaries:
        return pd.DataFrame()
    return pd.concat(summaries, ignore_index=True)


def month_diff(start_dates, end_dates):
    start = pd.to_datetime(start_dates, errors="coerce")
    end = pd.to_datetime(end_dates, errors="coerce")
    months = (end.dt.year - start.dt.year) * 12 + (end.dt.month - start.dt.month)
    return months.astype("float")


def scheduled_balance_proxy(funded_amount, installment, annual_rate, months_elapsed):
    principal = pd.Series(funded_amount, copy=False).astype(float).fillna(0)
    payment = pd.Series(installment, copy=False).astype(float).fillna(0)
    annual_rate = pd.Series(annual_rate, copy=False).astype(float).fillna(0)
    elapsed = pd.Series(months_elapsed, copy=False).astype(float).fillna(0).clip(lower=0)

    monthly_rate = annual_rate / 1200.0
    growth = np.power(1 + monthly_rate, elapsed)
    with np.errstate(divide="ignore", invalid="ignore"):
        amortized_balance = principal * growth - payment * ((growth - 1) / monthly_rate)

    zero_rate_mask = monthly_rate.abs() < 1e-9
    amortized_balance = np.where(
        zero_rate_mask,
        principal - payment * elapsed,
        amortized_balance,
    )
    return pd.Series(amortized_balance).fillna(0).clip(lower=0)


def _parse_percent_series(series):
    text_series = series.astype(str).str.replace("%", "", regex=False).str.strip()
    return pd.to_numeric(text_series, errors="coerce")


def _year_quarter_label(date_series):
    date_series = pd.to_datetime(date_series, errors="coerce")
    years = date_series.dt.year.astype("Int64").astype(str)
    quarters = date_series.dt.quarter.astype("Int64").astype(str)
    labels = years + "Q" + quarters
    return labels.where(date_series.notna(), other=pd.NA)


def _summarize_lgd_group(
    df,
    group_columns,
    mean_column_name,
    count_column_name,
    expected_column_name,
    portfolio_mean,
    shrinkage_k,
):
    grouped = (
        df.groupby(group_columns, dropna=False)["lgd_proxy"]
        .agg(["mean", "size"])
        .reset_index()
        .rename(columns={"mean": mean_column_name, "size": count_column_name})
    )
    grouped[f"{expected_column_name}_raw"] = grouped[mean_column_name]
    grouped[f"{expected_column_name}_weight"] = grouped[count_column_name] / (
        grouped[count_column_name] + shrinkage_k
    )
    grouped[expected_column_name] = (
        grouped[f"{expected_column_name}_weight"] * grouped[mean_column_name]
        + (1 - grouped[f"{expected_column_name}_weight"]) * portfolio_mean
    )
    grouped[f"{expected_column_name}_portfolio_mean"] = portfolio_mean

    if expected_column_name == "expected_lgd_exact":
        grouped = grouped.rename(
            columns={f"{expected_column_name}_weight": "exact_credibility_weight"}
        )
    return grouped
