import copy

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from evaluation import evaluate_model
from loss_preprocess import (
    ACTIVE_LOAN_STATUSES,
    RESOLVED_LOAN_STATUSES,
    SPLIT_LABEL_COLUMN,
    STATUS_COLUMN,
)


HAZARD_GROUP_COLUMNS = [
    "grade",
    "fico_bucket",
    "purpose_group",
    "annual_income_band",
    "term_months",
]
HAZARD_MODEL_COLUMNS = [
    "grade",
    "fico_bucket",
    "purpose_group",
    "annual_income_band",
    "term_months",
    "month_bucket",
    "term_month_interaction",
]
HAZARD_STATE_COLUMNS = HAZARD_GROUP_COLUMNS + ["start_month"]
MONTH_BUCKET_LABELS = [
    "m01_03",
    "m04_06",
    "m07_12",
    "m13_24",
    "m25_36",
    "m37_48",
    "m49_60",
]
DEFAULT_HAZARD_PARAM_GRID = [
    {"C": 0.5, "class_weight": None},
    {"C": 1.0, "class_weight": None},
    {"C": 2.0, "class_weight": "balanced"},
]
DEFAULT_HAZARD_BASE_PARAMS = {
    "solver": "lbfgs",
    "max_iter": 500,
    "n_jobs": None,
}


def prepare_hazard_loan_frame(df):
    prepared = df.copy()
    prepared = prepared.dropna(
        subset=[
            SPLIT_LABEL_COLUMN,
            "issue_date",
            "grade",
            "fico_bucket",
            "purpose_group",
            "annual_income_band",
            "term_months",
            "months_on_book",
        ]
    ).copy()
    prepared["term_months"] = prepared["term_months"].round().astype(int).clip(lower=1)
    prepared["observed_months"] = (
        prepared["months_on_book"].fillna(0).round().astype(int).clip(lower=1)
    )
    prepared["observed_months"] = np.minimum(
        prepared["observed_months"],
        prepared["term_months"],
    )
    prepared["event_default"] = (prepared[STATUS_COLUMN] == "Charged Off").astype(int)
    prepared["resolved_flag"] = prepared[STATUS_COLUMN].isin(RESOLVED_LOAN_STATUSES).astype(int)
    prepared["active_flag"] = prepared[STATUS_COLUMN].isin(ACTIVE_LOAN_STATUSES).astype(int)
    prepared["actual_default"] = prepared["event_default"]
    return prepared.reset_index(drop=True)


def build_monthly_hazard_panel_counts(loan_df, chunk_size=200000):
    group_columns = [SPLIT_LABEL_COLUMN] + HAZARD_MODEL_COLUMNS
    grouped_chunks = []

    working = loan_df[
        [SPLIT_LABEL_COLUMN, "observed_months", "event_default"] + HAZARD_GROUP_COLUMNS
    ].copy()

    for start in range(0, len(working), chunk_size):
        chunk = working.iloc[start : start + chunk_size].copy()
        durations = chunk["observed_months"].to_numpy(dtype=int)
        if len(durations) == 0:
            continue

        repeated_index = np.repeat(np.arange(len(chunk)), durations)
        repeated = chunk.iloc[repeated_index].reset_index(drop=True)

        duration_offsets = np.repeat(np.cumsum(durations) - durations, durations)
        month_on_book = np.arange(durations.sum()) - duration_offsets + 1
        repeated["month_on_book"] = month_on_book
        repeated["month_bucket"] = month_bucket_label(month_on_book)
        repeated["term_month_interaction"] = (
            repeated["term_months"].astype(int).astype(str)
            + "_"
            + repeated["month_bucket"].astype(str)
        )

        event_month = np.repeat(durations, durations)
        event_flag = np.repeat(chunk["event_default"].to_numpy(dtype=int), durations)
        repeated["event_indicator"] = (
            (repeated["month_on_book"].to_numpy() == event_month) & (event_flag == 1)
        ).astype(int)

        grouped_chunk = (
            repeated.groupby(group_columns, dropna=False)
            .agg(
                exposure_count=("event_indicator", "size"),
                event_count=("event_indicator", "sum"),
            )
            .reset_index()
        )
        grouped_chunks.append(grouped_chunk)

    if not grouped_chunks:
        return pd.DataFrame(columns=group_columns + ["exposure_count", "event_count"])

    return (
        pd.concat(grouped_chunks, ignore_index=True)
        .groupby(group_columns, dropna=False, as_index=False)
        .agg(
            exposure_count=("exposure_count", "sum"),
            event_count=("event_count", "sum"),
        )
        .reset_index(drop=True)
    )


def fit_pooled_logistic_hazard(loan_df, param_grid=None):
    hazard_loans = prepare_hazard_loan_frame(loan_df)
    panel_counts = build_monthly_hazard_panel_counts(hazard_loans)
    if panel_counts.empty:
        raise ValueError("Hazard panel construction produced no rows.")

    resolved_validation = hazard_loans[
        (hazard_loans[SPLIT_LABEL_COLUMN] == "validation")
        & (hazard_loans["resolved_flag"] == 1)
    ].copy()
    resolved_test = hazard_loans[
        (hazard_loans[SPLIT_LABEL_COLUMN] == "test")
        & (hazard_loans["resolved_flag"] == 1)
    ].copy()

    search_rows = []
    best_search_key = None
    best_candidate = None
    param_grid = param_grid or DEFAULT_HAZARD_PARAM_GRID

    for candidate_id, override in enumerate(param_grid, start=1):
        train_bundle = _fit_hazard_model(
            panel_counts[panel_counts[SPLIT_LABEL_COLUMN] == "train"].copy(),
            override,
        )
        validation_predictions = score_hazard_probabilities(
            train_bundle,
            resolved_validation,
            start_month_column=None,
        )
        validation_metrics = evaluate_model(
            validation_predictions["actual_default"],
            validation_predictions["lifetime_pd"],
        )
        validation_metrics_12m = evaluate_model(
            validation_predictions["actual_default_12m"],
            validation_predictions["pd_12m"],
        )
        comparison_key = _hazard_selection_key(validation_metrics)
        search_rows.append(
            {
                "candidate_id": candidate_id,
                "params": override,
                "validation_lifetime_auc": validation_metrics["AUC"],
                "validation_lifetime_ks": validation_metrics["KS"],
                "validation_lifetime_brier": validation_metrics["Brier"],
                "validation_12m_auc": validation_metrics_12m["AUC"],
                "validation_12m_brier": validation_metrics_12m["Brier"],
            }
        )
        if best_search_key is None or comparison_key > best_search_key:
            best_search_key = comparison_key
            best_candidate = {
                "params": copy.deepcopy(override),
                "train_bundle": train_bundle,
                "validation_predictions": validation_predictions,
                "validation_metrics": validation_metrics,
                "validation_metrics_12m": validation_metrics_12m,
            }

    search_table = (
        pd.DataFrame(search_rows)
        .sort_values(
            by=[
                "validation_lifetime_auc",
                "validation_lifetime_ks",
                "validation_lifetime_brier",
            ],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )
    search_table["validation_rank"] = range(1, len(search_table) + 1)

    final_training_panel = panel_counts[
        panel_counts[SPLIT_LABEL_COLUMN].isin(["train", "validation"])
    ].copy()
    final_bundle = _fit_hazard_model(final_training_panel, best_candidate["params"])
    test_predictions = score_hazard_probabilities(
        final_bundle,
        resolved_test,
        start_month_column=None,
    )
    test_metrics = evaluate_model(
        test_predictions["actual_default"],
        test_predictions["lifetime_pd"],
    )
    test_metrics_12m = evaluate_model(
        test_predictions["actual_default_12m"],
        test_predictions["pd_12m"],
    )

    panel_summary = (
        panel_counts.groupby(SPLIT_LABEL_COLUMN, as_index=False)
        .agg(
            panel_cells=("exposure_count", "size"),
            exposure_count=("exposure_count", "sum"),
            event_count=("event_count", "sum"),
        )
        .reset_index(drop=True)
    )

    return {
        "hazard_loans": hazard_loans,
        "panel_counts": panel_counts,
        "panel_summary": panel_summary,
        "search_table": search_table,
        "selected_params": best_candidate["params"],
        "train_bundle": best_candidate["train_bundle"],
        "final_bundle": final_bundle,
        "validation_predictions": best_candidate["validation_predictions"],
        "validation_metrics": best_candidate["validation_metrics"],
        "validation_metrics_12m": best_candidate["validation_metrics_12m"],
        "validation_vintage_table": build_hazard_vintage_calibration(
            best_candidate["validation_predictions"],
            split_name="validation",
        ),
        "test_predictions": test_predictions,
        "test_metrics": test_metrics,
        "test_metrics_12m": test_metrics_12m,
        "test_vintage_table": build_hazard_vintage_calibration(
            test_predictions,
            split_name="test",
        ),
    }


def score_hazard_probabilities(model_bundle, loan_df, start_month_column="months_on_book"):
    scoring = loan_df.copy()
    if scoring.empty:
        return scoring.assign(pd_12m=pd.Series(dtype=float), lifetime_pd=pd.Series(dtype=float))

    if start_month_column is None:
        scoring["start_month"] = 0
    else:
        scoring["start_month"] = (
            scoring[start_month_column].fillna(0).round().astype(int).clip(lower=0)
        )
    scoring["term_months"] = scoring["term_months"].round().astype(int).clip(lower=1)
    scoring["start_month"] = np.minimum(scoring["start_month"], scoring["term_months"])

    state_table = scoring[HAZARD_STATE_COLUMNS].drop_duplicates().reset_index(drop=True)
    state_table["state_id"] = np.arange(len(state_table))
    future_rows = _build_future_state_rows(state_table)

    if future_rows.empty:
        state_scores = state_table[HAZARD_STATE_COLUMNS + ["state_id"]].copy()
        state_scores["pd_12m"] = 0.0
        state_scores["lifetime_pd"] = 0.0
    else:
        design_matrix = build_hazard_design_matrix(
            future_rows[HAZARD_MODEL_COLUMNS],
            design_columns=model_bundle["design_columns"],
        )
        future_rows["hazard_prob"] = model_bundle["model"].predict_proba(design_matrix)[:, 1]
        state_scores = _aggregate_state_probabilities(future_rows)

    scoring = scoring.merge(
        state_scores[HAZARD_STATE_COLUMNS + ["pd_12m", "lifetime_pd"]],
        on=HAZARD_STATE_COLUMNS,
        how="left",
    )
    scoring["pd_12m"] = scoring["pd_12m"].fillna(0).clip(lower=0, upper=1)
    scoring["lifetime_pd"] = scoring["lifetime_pd"].fillna(0).clip(lower=0, upper=1)
    return scoring


def build_hazard_vintage_calibration(prediction_df, split_name):
    if prediction_df.empty:
        return pd.DataFrame()

    summary = (
        prediction_df.groupby("issue_year", as_index=False)
        .agg(
            loan_count=("actual_default", "size"),
            actual_default_rate=("actual_default", "mean"),
            predicted_lifetime_pd=("lifetime_pd", "mean"),
            actual_default_12m_rate=("actual_default_12m", "mean"),
            predicted_12m_pd=("pd_12m", "mean"),
        )
        .reset_index(drop=True)
    )
    summary["split_name"] = split_name
    return summary


def build_hazard_design_matrix(df, design_columns=None):
    design = df.copy()
    design["term_months"] = design["term_months"].astype(int).astype(str)
    design = pd.get_dummies(design[HAZARD_MODEL_COLUMNS], drop_first=True, dtype=float)
    if design_columns is None:
        return design

    aligned = design.copy()
    for column in design_columns:
        if column not in aligned.columns:
            aligned[column] = 0.0
    aligned = aligned.reindex(columns=design_columns, fill_value=0.0)
    return aligned


def month_bucket_label(month_values):
    month_array = pd.Series(month_values, copy=False).astype(int)
    return pd.Series(
        np.select(
            [
                month_array <= 3,
                month_array <= 6,
                month_array <= 12,
                month_array <= 24,
                month_array <= 36,
                month_array <= 48,
            ],
            MONTH_BUCKET_LABELS[:-1],
            default=MONTH_BUCKET_LABELS[-1],
        ),
        index=month_array.index,
    )


def _fit_hazard_model(panel_counts, override_params):
    model_frame = _expand_panel_counts_for_logistic(panel_counts)
    design_matrix = build_hazard_design_matrix(model_frame[HAZARD_MODEL_COLUMNS])

    model_params = copy.deepcopy(DEFAULT_HAZARD_BASE_PARAMS)
    model_params.update(override_params)
    model = LogisticRegression(**model_params)
    model.fit(
        design_matrix,
        model_frame["target"],
        sample_weight=model_frame["sample_weight"],
    )
    return {
        "model": model,
        "design_columns": list(design_matrix.columns),
        "model_params": model_params,
    }


def _expand_panel_counts_for_logistic(panel_counts):
    positive = panel_counts[panel_counts["event_count"] > 0].copy()
    positive["target"] = 1
    positive["sample_weight"] = positive["event_count"]

    negative = panel_counts[(panel_counts["exposure_count"] - panel_counts["event_count"]) > 0].copy()
    negative["target"] = 0
    negative["sample_weight"] = negative["exposure_count"] - negative["event_count"]

    return pd.concat([positive, negative], ignore_index=True)


def _build_future_state_rows(state_table):
    repeated_rows = []
    for row in state_table.itertuples(index=False):
        if row.start_month >= row.term_months:
            continue
        future_months = np.arange(int(row.start_month) + 1, int(row.term_months) + 1)
        repeated_rows.append(
            pd.DataFrame(
                {
                    "state_id": row.state_id,
                    "grade": row.grade,
                    "fico_bucket": row.fico_bucket,
                    "purpose_group": row.purpose_group,
                    "annual_income_band": row.annual_income_band,
                    "term_months": int(row.term_months),
                    "start_month": int(row.start_month),
                    "future_month": future_months,
                }
            )
        )

    if not repeated_rows:
        return pd.DataFrame(
            columns=[
                "state_id",
                "grade",
                "fico_bucket",
                "purpose_group",
                "annual_income_band",
                "term_months",
                "start_month",
                "future_month",
                "month_bucket",
                "term_month_interaction",
            ]
        )

    future_rows = pd.concat(repeated_rows, ignore_index=True)
    future_rows["month_bucket"] = month_bucket_label(future_rows["future_month"])
    future_rows["term_month_interaction"] = (
        future_rows["term_months"].astype(int).astype(str)
        + "_"
        + future_rows["month_bucket"].astype(str)
    )
    return future_rows


def _aggregate_state_probabilities(future_rows):
    ordered = future_rows.sort_values(["state_id", "future_month"]).reset_index(drop=True)
    state_rows = []
    for state_id, group in ordered.groupby("state_id", sort=False):
        hazards = group["hazard_prob"].to_numpy(dtype=float)
        pd_12m = 1 - np.prod(1 - hazards[:12]) if len(hazards) else 0.0
        lifetime_pd = 1 - np.prod(1 - hazards) if len(hazards) else 0.0
        base = group.iloc[0][HAZARD_STATE_COLUMNS].to_dict()
        base["state_id"] = state_id
        base["pd_12m"] = float(np.clip(pd_12m, 0, 1))
        base["lifetime_pd"] = float(np.clip(lifetime_pd, 0, 1))
        state_rows.append(base)

    return pd.DataFrame(state_rows)


def _hazard_selection_key(validation_metrics):
    return (
        float(validation_metrics["AUC"]),
        float(validation_metrics["KS"]),
        -float(validation_metrics["Brier"]),
    )
