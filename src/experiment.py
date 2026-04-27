import copy
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid, train_test_split

from evaluation import evaluate_model
from loss_preprocess import (
    ACTIVE_LOAN_STATUSES,
    ANNUAL_INCOME_BINS,
    ANNUAL_INCOME_LABELS,
    LOSS_STATUS_VALUES,
    RESOLVED_LOAN_STATUSES,
    month_diff,
    scheduled_balance_proxy,
)
from model_HGB import build_hist_gradient_boosting_model
from model_LR import build_logistic_model
from model_MLP import build_mlp_model
from model_RF import build_random_forest_model
from model_XGB import build_xgboost_model
from preprocess import (
    ADDITIONAL_NUMERIC_FEATURE_COLUMNS,
    BASE_ENGINEERED_FEATURE_COLUMNS,
    BASE_FEATURE_COLUMNS,
    DTI_UPPER_BOUND,
    EXTENDED_ENGINEERED_FEATURE_COLUMNS,
    FEATURE_FLAG_CANDIDATES,
    FICO_BUCKET_BINS,
    FICO_BUCKET_LABELS,
    HIGH_DTI_THRESHOLD,
    HIGH_REVOL_UTIL_THRESHOLD,
    ID_COLUMN,
    ISSUE_DATE_COLUMN,
    ISSUE_RAW_COLUMN,
    RATE_UPPER_BOUND,
    RECENT_INQUIRY_THRESHOLD,
    REVOL_UTIL_UPPER_BOUND,
    STATUS_COLUMN,
    TARGET_COLUMN,
    ensure_model_ready_dataset,
    ensure_sample_id,
    parse_emp_length_value,
    reconstruct_raw_like_dataset,
    select_feature_columns,
)
from split import (
    DEFAULT_TEMPORAL_SPLIT,
    assign_temporal_split_labels,
    combine_training_frames,
    temporal_train_valid_test_split,
)
from hazard import build_monthly_hazard_panel_counts


MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "mlp_neural_network": "MLP Neural Network",
    "hist_gradient_boosting": "HistGradientBoosting",
}
MODEL_BUILDERS = {
    "logistic_regression": build_logistic_model,
    "random_forest": build_random_forest_model,
    "xgboost": build_xgboost_model,
    "mlp_neural_network": build_mlp_model,
    "hist_gradient_boosting": build_hist_gradient_boosting_model,
}

PD_MODELING_TYPE_HAZARD = "hazard"
PD_MODELING_TYPE_CALENDAR_TIME_HAZARD = "calendar_time_hazard"
PD_MODELING_TYPE_STATIC = "static"
COMPACT_HAZARD_PROFILE = "compact_hazard"
FULL_HAZARD_PROFILE = "full_hazard"
COMPACT_CALENDAR_HAZARD_PROFILE = "compact_calendar_hazard"
FULL_CALENDAR_HAZARD_PROFILE = "full_calendar_hazard"
GROUPED_COMPACT_HAZARD_PROFILE = "grouped_compact_hazard"
GROUPED_FULL_HAZARD_PROFILE = "grouped_full_hazard"
GROUPED_HAZARD_STATE_COLUMNS = [
    "grade",
    "fico_bucket",
    "purpose_group",
    "annual_income_band",
    "term_months",
]

LAST_PAYMENT_RAW_COLUMN = "last_pymnt_d"
LAST_PAYMENT_DATE_COLUMN = "last_payment_date"
SNAPSHOT_MONTH_COLUMN = "snapshot_month"
TARGET_1M_COLUMN = "target_1m"
DEFAULT_CALENDAR_DATA_CUTOFF = "2018-12-01"
SPLIT_LABEL_COLUMN = "split_label"
OBSERVED_MONTHS_COLUMN = "observed_months"

HAZARD_MONTH_BUCKET_LABELS = [
    "m01_03",
    "m04_06",
    "m07_12",
    "m13_24",
    "m25_36",
    "m37_48",
    "m49_60",
]
HAZARD_NUMERIC_COLUMNS = list(
    dict.fromkeys(
        BASE_FEATURE_COLUMNS
        + ADDITIONAL_NUMERIC_FEATURE_COLUMNS
        + BASE_ENGINEERED_FEATURE_COLUMNS
        + EXTENDED_ENGINEERED_FEATURE_COLUMNS
    )
)
HAZARD_CATEGORICAL_COLUMNS = [
    "home_ownership",
    "purpose",
    "fico_bucket",
    "verification_status",
]
HAZARD_LEAKAGE_COLUMNS = {
    LAST_PAYMENT_RAW_COLUMN,
    LAST_PAYMENT_DATE_COLUMN,
    "next_pymnt_d",
    STATUS_COLUMN,
    "recoveries",
    "total_pymnt",
    "total_pymnt_inv",
    "total_rec_prncp",
    "total_rec_int",
    "out_prncp",
    "out_prncp_inv",
    "collection_recovery_fee",
    "last_pymnt_amnt",
}
DEFAULT_REDUCED_PARAM_GRID = {
    "logistic_regression": {
        "C": [0.25, 0.5, 1.0, 2.0],
        "class_weight": [None, "balanced"],
    },
    "random_forest": {
        "n_estimators": [120, 180],
        "max_depth": [10, 14, 18],
        "min_samples_leaf": [15, 25, 40],
    },
    "xgboost": {
        "n_estimators": [80, 120, 160],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.03, 0.05, 0.08],
        "max_candidates": 24,
    },
    "mlp_neural_network": {
        "hidden_layer_sizes": [(32,), (64,), (64, 32)],
        "alpha": [0.0001, 0.001],
        "learning_rate_init": [0.001, 0.003],
    },
}
HISTORIC_STRONG_PARAM_CANDIDATES = {
    "xgboost": {
        "n_estimators": 160,
        "max_depth": 6,
        "learning_rate": 0.08,
    },
}


def _default_enabled_model_keys():
    return [
        "logistic_regression",
        "random_forest",
        "xgboost",
        "mlp_neural_network",
    ]


def prepare_experiment_data(config, df):
    modeling_type = config.get("modeling_type", PD_MODELING_TYPE_HAZARD)
    if modeling_type == PD_MODELING_TYPE_STATIC:
        return _prepare_static_experiment_data(config, df)
    if modeling_type == PD_MODELING_TYPE_CALENDAR_TIME_HAZARD:
        return prepare_calendar_time_hazard_experiment_data(config, df)
    return prepare_hazard_experiment_data(config, df)


def _prepare_static_experiment_data(config, df):
    working = ensure_model_ready_dataset(df)
    temporal_split = copy.deepcopy(DEFAULT_TEMPORAL_SPLIT)
    temporal_split.update(config.get("temporal_split", {}))

    feature_columns = select_feature_columns(
        working,
        config.get("feature_flags"),
    )
    if not feature_columns:
        raise ValueError("No feature columns were selected by the current feature flags.")

    meta_columns = [
        column
        for column in config.get(
            "meta_columns",
            [
                ID_COLUMN,
                ISSUE_DATE_COLUMN,
                "loan_amnt",
                "annual_inc",
                "fico_range_low",
                "term_months",
            ],
        )
        if column in working.columns
    ]

    selected_columns = list(dict.fromkeys(meta_columns + feature_columns + [TARGET_COLUMN]))
    experiment_df = working[selected_columns].copy()

    split_payload = temporal_train_valid_test_split(
        experiment_df,
        feature_columns=feature_columns,
        target_col=TARGET_COLUMN,
        date_col=ISSUE_DATE_COLUMN,
        meta_columns=meta_columns,
        train_end=temporal_split["train_end"],
        valid_end=temporal_split["valid_end"],
        sample_frac=float(config.get("sample_frac", 1.0)),
        random_state=int(config.get("random_state", 42)),
    )

    return {
        "df": experiment_df,
        "feature_columns": feature_columns,
        "meta_columns": meta_columns,
        "train": split_payload["train"],
        "validation": split_payload["validation"],
        "test": split_payload["test"],
        "split_summary": split_payload["split_summary"],
        "temporal_bounds": split_payload["temporal_bounds"],
    }


def prepare_hazard_experiment_data(config, df):
    temporal_split = copy.deepcopy(DEFAULT_TEMPORAL_SPLIT)
    temporal_split.update(config.get("temporal_split", {}))
    hazard_feature_mode = config.get("hazard_feature_mode", "loan_month")

    loan_frame = prepare_pd_hazard_loan_frame(
        df,
        train_end=temporal_split["train_end"],
        valid_end=temporal_split["valid_end"],
    )
    loan_frame = _apply_loan_level_sampling(
        loan_frame,
        sample_frac=float(config.get("sample_frac", 1.0)),
        random_state=int(config.get("random_state", 42)),
    )
    loan_frame, impute_values = _fit_apply_hazard_imputation(loan_frame)

    meta_columns = [
        column
        for column in config.get(
            "meta_columns",
            [
                ID_COLUMN,
                ISSUE_DATE_COLUMN,
                "loan_amnt",
                "annual_inc",
                "fico_range_low",
                "term_months",
                "int_rate",
                "revol_util",
                "months_on_book",
                "remaining_term",
            ],
        )
        if column in loan_frame.columns
    ]

    partitions = {
        "train": loan_frame[loan_frame[SPLIT_LABEL_COLUMN] == "train"].copy(),
        "validation": loan_frame[loan_frame[SPLIT_LABEL_COLUMN] == "validation"].copy(),
        "test": loan_frame[loan_frame[SPLIT_LABEL_COLUMN] == "test"].copy(),
    }
    for name, partition in partitions.items():
        if partition.empty:
            raise ValueError(
                f"The temporal {name} partition is empty for hazard modeling. "
                "Use data with issue_date, loan_status, term, and observed months, or adjust the temporal boundaries."
            )

    feature_columns_by_profile = {}
    panel_payloads_by_profile = {}
    grouped_train_counts = (
        _build_grouped_hazard_panel_counts(partitions["train"])
        if hazard_feature_mode == "grouped"
        else None
    )
    profile_candidates = (
        [GROUPED_COMPACT_HAZARD_PROFILE, GROUPED_FULL_HAZARD_PROFILE]
        if hazard_feature_mode == "grouped"
        else [COMPACT_HAZARD_PROFILE, FULL_HAZARD_PROFILE]
    )
    for profile in profile_candidates:
        panel = _build_hazard_panel_payload(
            partitions["train"],
            feature_profile=profile,
            meta_columns=meta_columns,
            panel_counts=grouped_train_counts,
        )
        panel_payloads_by_profile[profile] = panel
        feature_columns_by_profile[profile] = panel["feature_columns"]

    feature_columns_by_model = {
        model_name: feature_columns_by_profile[
            _hazard_feature_profile_for_model(model_name, hazard_feature_mode)
        ]
        for model_name in config.get("enabled_models", MODEL_LABELS.keys())
    }

    split_summary = pd.DataFrame(
        [
            _hazard_partition_summary("train", partitions["train"]),
            _hazard_partition_summary("validation", partitions["validation"]),
            _hazard_partition_summary("test", partitions["test"]),
        ]
    )

    train_payload = _hazard_loan_payload(partitions["train"], meta_columns)
    if panel_payloads_by_profile:
        train_payload["hazard_panel_payloads_by_profile"] = panel_payloads_by_profile

    return {
        "modeling_type": PD_MODELING_TYPE_HAZARD,
        "hazard_feature_mode": hazard_feature_mode,
        "df": loan_frame.copy(),
        "feature_columns": feature_columns_by_profile[profile_candidates[-1]],
        "feature_columns_by_profile": feature_columns_by_profile,
        "feature_columns_by_model": feature_columns_by_model,
        "feature_profiles_by_model": {
            model_name: _hazard_feature_profile_for_model(model_name, hazard_feature_mode)
            for model_name in config.get("enabled_models", MODEL_LABELS.keys())
        },
        "hazard_impute_values": impute_values,
        "meta_columns": meta_columns,
        "train": train_payload,
        "validation": _hazard_loan_payload(partitions["validation"], meta_columns),
        "test": _hazard_loan_payload(partitions["test"], meta_columns),
        "split_summary": split_summary,
        "temporal_bounds": {
            "train_end": pd.Timestamp(temporal_split["train_end"]),
            "valid_end": pd.Timestamp(temporal_split["valid_end"]),
        },
    }


def prepare_calendar_time_hazard_experiment_data(config, df):
    temporal_split = copy.deepcopy(DEFAULT_TEMPORAL_SPLIT)
    temporal_split.update(config.get("temporal_split", {}))
    data_cutoff = pd.Timestamp(config.get("data_cutoff", DEFAULT_CALENDAR_DATA_CUTOFF))
    data_cutoff = data_cutoff.to_period("M").to_timestamp()

    if config.get("calendar_panel_is_prebuilt", False):
        panel = df.copy()
        for column in [SNAPSHOT_MONTH_COLUMN, ISSUE_DATE_COLUMN, LAST_PAYMENT_DATE_COLUMN]:
            if column in panel.columns:
                panel[column] = pd.to_datetime(panel[column], errors="coerce")
        if "target_12m" not in panel.columns:
            panel = _add_calendar_time_features(panel)
        for column in [TARGET_1M_COLUMN, "actual_default", "actual_default_12m"]:
            if column in panel.columns:
                panel[column] = pd.to_numeric(panel[column], errors="coerce")
    else:
        loan_frame = prepare_pd_hazard_loan_frame(
            df,
            impute_values=None,
            require_outcomes=True,
        )
        panel = _build_calendar_survival_panel_frame(
            loan_frame,
            data_cutoff=data_cutoff,
            train_end=temporal_split["train_end"],
            valid_end=temporal_split["valid_end"],
            include_scoring_snapshot=bool(config.get("include_scoring_snapshot", True)),
        )
        panel = _apply_calendar_panel_sampling(
            panel,
            sample_frac=float(config.get("sample_frac", 1.0)),
            random_state=int(config.get("random_state", 42)),
        )
    panel, impute_values = _fit_apply_calendar_hazard_imputation(panel)

    output_path = config.get("calendar_panel_path")
    counts_path = config.get("calendar_panel_counts_path")
    if output_path:
        _write_frame_with_parquet_fallback(panel, output_path)
    panel_counts = _build_calendar_panel_counts(panel)
    if counts_path:
        _write_frame_with_parquet_fallback(panel_counts, counts_path)

    meta_columns = [
        column
        for column in config.get(
            "meta_columns",
            [
                ID_COLUMN,
                SNAPSHOT_MONTH_COLUMN,
                ISSUE_DATE_COLUMN,
                "loan_amnt",
                "annual_inc",
                "fico_range_low",
                "term_months",
                "int_rate",
                "revol_util",
                "month_on_book",
                "remaining_term",
                TARGET_1M_COLUMN,
            ],
        )
        if column in panel.columns
    ]

    model_panel = panel[panel[SPLIT_LABEL_COLUMN].isin(["train", "validation", "test"])].copy()
    partitions = {
        "train": model_panel[model_panel[SPLIT_LABEL_COLUMN] == "train"].copy(),
        "validation": model_panel[model_panel[SPLIT_LABEL_COLUMN] == "validation"].copy(),
        "test": model_panel[model_panel[SPLIT_LABEL_COLUMN] == "test"].copy(),
    }
    for name, partition in partitions.items():
        if partition.empty:
            raise ValueError(
                f"The calendar-time {name} partition is empty. Adjust snapshot split bounds "
                "or provide raw loan data with issue_d, loan_status, last_pymnt_d, and term."
            )

    enabled_models = config.get("enabled_models", _default_enabled_model_keys())
    profile_candidates = [
        COMPACT_CALENDAR_HAZARD_PROFILE,
        FULL_CALENDAR_HAZARD_PROFILE,
    ]
    feature_columns_by_profile = {}
    for profile in profile_candidates:
        payload = _build_calendar_hazard_panel_payload(
            partitions["train"],
            feature_profile=profile,
            meta_columns=meta_columns,
        )
        feature_columns_by_profile[profile] = payload["feature_columns"]

    feature_columns_by_model = {
        model_name: feature_columns_by_profile[
            _hazard_feature_profile_for_model(model_name, "calendar_time")
        ]
        for model_name in enabled_models
    }

    split_summary = pd.DataFrame(
        [
            _calendar_partition_summary("train", partitions["train"]),
            _calendar_partition_summary("validation", partitions["validation"]),
            _calendar_partition_summary("test", partitions["test"]),
        ]
    )

    return {
        "modeling_type": PD_MODELING_TYPE_CALENDAR_TIME_HAZARD,
        "hazard_feature_mode": "calendar_time",
        "data_cutoff": data_cutoff,
        "df": panel.copy(),
        "panel_counts": panel_counts,
        "feature_columns": feature_columns_by_profile[FULL_CALENDAR_HAZARD_PROFILE],
        "feature_columns_by_profile": feature_columns_by_profile,
        "feature_columns_by_model": feature_columns_by_model,
        "feature_profiles_by_model": {
            model_name: _hazard_feature_profile_for_model(model_name, "calendar_time")
            for model_name in enabled_models
        },
        "hazard_impute_values": impute_values,
        "meta_columns": meta_columns,
        "train": _calendar_panel_payload(partitions["train"], meta_columns),
        "validation": _calendar_panel_payload(partitions["validation"], meta_columns),
        "test": _calendar_panel_payload(partitions["test"], meta_columns),
        "scoring": _calendar_panel_payload(
            panel[panel[SPLIT_LABEL_COLUMN] == "scoring"].copy(),
            meta_columns,
        ),
        "split_summary": split_summary,
        "temporal_bounds": {
            "train_end": pd.Timestamp(temporal_split["train_end"]),
            "valid_end": pd.Timestamp(temporal_split["valid_end"]),
            "data_cutoff": data_cutoff,
        },
    }


def run_experiment(config, df):
    prepared = prepare_experiment_data(config, df)
    if prepared.get("modeling_type") == PD_MODELING_TYPE_CALENDAR_TIME_HAZARD:
        return _run_calendar_time_hazard_experiment(config, prepared)
    if prepared.get("modeling_type") == PD_MODELING_TYPE_HAZARD:
        return _run_hazard_experiment(config, prepared)

    train_payload = prepared["train"]
    validation_payload = prepared["validation"]
    test_payload = prepared["test"]
    train_validation_payload = combine_training_frames(train_payload, validation_payload)

    validation_rows = []
    validation_predictions = {}
    train_only_models = {}
    best_params_by_model = {}
    grid_search_tables = {}

    for model_name in config.get("enabled_models", []):
        search_result = _search_model(
            model_name=model_name,
            config=config,
            train_payload=train_payload,
            validation_payload=validation_payload,
        )
        best_params = search_result["best_params"]
        best_params_by_model[model_name] = best_params
        grid_search_tables[model_name] = search_result["search_table"]
        train_only_models[model_name] = search_result["best_model"]
        validation_predictions[model_name] = search_result["prediction_frame"]

        best_row = search_result["best_row"]
        validation_rows.append(
            {
                "model_key": model_name,
                "model_name": MODEL_LABELS[model_name],
                "params": json.dumps(best_params, sort_keys=True),
                "feature_count": len(prepared["feature_columns"]),
                "search_mode": search_result["search_mode"],
                "search_candidates": search_result["search_candidates"],
                "validation_auc": best_row["validation_auc"],
                "validation_ks": best_row["validation_ks"],
                "validation_brier": best_row["validation_brier"],
                "calibration_summary": best_row["calibration_summary"],
                "validation_metrics": best_row["validation_metrics"],
            }
        )

    validation_table = rank_validation_results(pd.DataFrame(validation_rows))
    validation_table["validation_rank"] = range(1, len(validation_table) + 1)
    champion_row = validation_table.iloc[0].to_dict()
    champion_key = champion_row["model_key"]

    final_test_rows = []
    final_models = {}
    test_predictions = {}
    for model_name in config.get("enabled_models", []):
        best_params = best_params_by_model[model_name]
        final_model = _build_model(model_name, best_params)
        final_model.fit(train_validation_payload["X"], train_validation_payload["y"])
        final_models[model_name] = final_model

        test_pred = final_model.predict_proba(test_payload["X"])[:, 1]
        test_metrics = evaluate_model(test_payload["y"], test_pred)
        test_predictions[model_name] = _build_prediction_frame(
            test_payload["meta"],
            test_payload["y"],
            test_pred,
            model_name,
        )

        validation_rank = int(
            validation_table.loc[
                validation_table["model_key"] == model_name,
                "validation_rank",
            ].iloc[0]
        )
        final_test_rows.append(
            {
                "model_key": model_name,
                "model_name": MODEL_LABELS[model_name],
                "validation_rank": validation_rank,
                "params": json.dumps(best_params, sort_keys=True),
                "search_mode": "refit_on_train_plus_validation",
                "test_auc": test_metrics["AUC"],
                "test_ks": test_metrics["KS"],
                "test_brier": test_metrics["Brier"],
                "calibration_summary": test_metrics["calibration_summary"],
                "test_metrics": test_metrics,
            }
        )

    final_test_table = (
        pd.DataFrame(final_test_rows)
        .sort_values("validation_rank")
        .reset_index(drop=True)
    )
    champion_test_row = final_test_table.loc[
        final_test_table["model_key"] == champion_key
    ].iloc[0].to_dict()

    champion_period_table = _build_champion_period_table(
        champion_key,
        train_payload,
        validation_payload,
        test_payload,
        train_only_models[champion_key],
        final_models[champion_key],
    )

    importance_tables = _build_importance_tables(
        final_models=final_models,
        feature_columns=prepared["feature_columns"],
    )

    combined_grid_search_table = _combine_search_tables(grid_search_tables)

    return {
        "validation_table": validation_table,
        "final_test_table": final_test_table,
        "champion_validation": champion_row,
        "champion_test": champion_test_row,
        "champion_period_table": champion_period_table,
        "champion_predictions": test_predictions[champion_key],
        "validation_predictions_by_model": validation_predictions,
        "test_predictions_by_model": test_predictions,
        "train_only_models": train_only_models,
        "final_models": final_models,
        "best_params_by_model": best_params_by_model,
        "grid_search_tables": grid_search_tables,
        "grid_search_table": combined_grid_search_table,
        "prepared_data": prepared,
        "importance_tables": importance_tables,
        "selected_model_key": champion_key,
    }


def _run_hazard_experiment(config, prepared):
    validation_rows = []
    validation_predictions = {}
    train_only_models = {}
    best_params_by_model = {}
    grid_search_tables = {}

    for model_name in config.get("enabled_models", []):
        search_result = _search_model(
            model_name=model_name,
            config=config,
            train_payload=prepared["train"],
            validation_payload=prepared["validation"],
        )
        best_params = search_result["best_params"]
        best_params_by_model[model_name] = best_params
        grid_search_tables[model_name] = search_result["search_table"]
        train_only_models[model_name] = search_result["best_model"]
        validation_predictions[model_name] = search_result["prediction_frame"]

        best_row = search_result["best_row"]
        validation_rows.append(
            {
                "model_key": model_name,
                "model_name": MODEL_LABELS[model_name],
                "params": json.dumps(best_params, sort_keys=True),
                "feature_profile": search_result["feature_profile"],
                "feature_count": search_result["feature_count"],
                "search_mode": search_result["search_mode"],
                "search_candidates": search_result["search_candidates"],
                "validation_auc": best_row["validation_auc"],
                "validation_ks": best_row["validation_ks"],
                "validation_brier": best_row["validation_brier"],
                "validation_12m_auc": best_row["validation_12m_auc"],
                "validation_12m_brier": best_row["validation_12m_brier"],
                "calibration_summary": best_row["calibration_summary"],
                "validation_metrics": best_row["validation_metrics"],
                "validation_metrics_12m": best_row["validation_metrics_12m"],
            }
        )

    validation_table = rank_validation_results(pd.DataFrame(validation_rows))
    validation_table["validation_rank"] = range(1, len(validation_table) + 1)
    champion_row = validation_table.iloc[0].to_dict()
    champion_key = champion_row["model_key"]

    final_models = {}
    test_predictions = {}
    final_test_rows = []
    train_validation_loans = pd.concat(
        [prepared["train"]["df"], prepared["validation"]["df"]],
        ignore_index=True,
    )
    final_panel_payloads_by_profile = {}
    if prepared.get("hazard_feature_mode") == "grouped":
        grouped_train_validation_counts = _build_grouped_hazard_panel_counts(
            train_validation_loans
        )
        for profile in set(prepared.get("feature_profiles_by_model", {}).values()):
            final_panel_payloads_by_profile[profile] = _build_hazard_panel_payload(
                train_validation_loans,
                feature_profile=profile,
                meta_columns=prepared["meta_columns"],
                panel_counts=grouped_train_validation_counts,
            )

    for model_name in config.get("enabled_models", []):
        feature_mode = prepared.get("hazard_feature_mode", "loan_month")
        feature_profile = _hazard_feature_profile_for_model(model_name, feature_mode)
        if feature_profile in final_panel_payloads_by_profile:
            final_model = _fit_hazard_model_from_payload(
                model_name,
                best_params_by_model[model_name],
                final_panel_payloads_by_profile[feature_profile],
                prepared["meta_columns"],
                feature_mode=feature_mode,
            )
        else:
            final_model = _fit_hazard_model_bundle(
                model_name,
                best_params_by_model[model_name],
                train_validation_loans,
                prepared["meta_columns"],
                feature_mode=feature_mode,
            )
        final_model.update(
            {
                "hazard_impute_values": prepared["hazard_impute_values"],
                "prepared_data": prepared,
            }
        )
        final_models[model_name] = final_model

        test_prediction = _score_hazard_model_bundle(
            final_model,
            prepared["test"]["df"],
            start_month_column=None,
        )
        test_metrics = evaluate_model(
            test_prediction["actual_default"],
            test_prediction["predicted_pd"],
        )
        test_metrics_12m = evaluate_model(
            test_prediction["actual_default_12m"],
            test_prediction["predicted_pd_12m"],
        )
        test_predictions[model_name] = test_prediction

        validation_rank = int(
            validation_table.loc[
                validation_table["model_key"] == model_name,
                "validation_rank",
            ].iloc[0]
        )
        final_test_rows.append(
            {
                "model_key": model_name,
                "model_name": MODEL_LABELS[model_name],
                "validation_rank": validation_rank,
                "params": json.dumps(best_params_by_model[model_name], sort_keys=True),
                "feature_profile": final_model["feature_profile"],
                "feature_count": len(final_model["feature_columns"]),
                "search_mode": "refit_on_train_plus_validation",
                "test_auc": test_metrics["AUC"],
                "test_ks": test_metrics["KS"],
                "test_brier": test_metrics["Brier"],
                "test_12m_auc": test_metrics_12m["AUC"],
                "test_12m_brier": test_metrics_12m["Brier"],
                "calibration_summary": test_metrics["calibration_summary"],
                "test_metrics": test_metrics,
                "test_metrics_12m": test_metrics_12m,
            }
        )

    final_test_table = (
        pd.DataFrame(final_test_rows)
        .sort_values("validation_rank")
        .reset_index(drop=True)
    )
    champion_test_row = final_test_table.loc[
        final_test_table["model_key"] == champion_key
    ].iloc[0].to_dict()

    champion_period_table = _build_hazard_champion_period_table(
        champion_key,
        prepared,
        train_only_models[champion_key],
        final_models[champion_key],
    )

    importance_tables = _build_importance_tables(
        final_models=final_models,
        feature_columns=prepared["feature_columns"],
    )
    combined_grid_search_table = _combine_search_tables(grid_search_tables)

    return {
        "validation_table": validation_table,
        "final_test_table": final_test_table,
        "champion_validation": champion_row,
        "champion_test": champion_test_row,
        "champion_period_table": champion_period_table,
        "champion_predictions": test_predictions[champion_key],
        "validation_predictions_by_model": validation_predictions,
        "test_predictions_by_model": test_predictions,
        "train_only_models": train_only_models,
        "final_models": final_models,
        "best_params_by_model": best_params_by_model,
        "grid_search_tables": grid_search_tables,
        "grid_search_table": combined_grid_search_table,
        "prepared_data": prepared,
        "importance_tables": importance_tables,
        "selected_model_key": champion_key,
    }


def _run_calendar_time_hazard_experiment(config, prepared):
    validation_rows = []
    validation_predictions = {}
    train_only_models = {}
    best_params_by_model = {}
    grid_search_tables = {}

    for model_name in config.get("enabled_models", []):
        search_result = _search_calendar_time_hazard_model(
            model_name=model_name,
            config=config,
            train_payload=prepared["train"],
            validation_payload=prepared["validation"],
        )
        best_params = search_result["best_params"]
        best_params_by_model[model_name] = best_params
        grid_search_tables[model_name] = search_result["search_table"]
        train_only_models[model_name] = search_result["best_model"]
        validation_predictions[model_name] = search_result["prediction_frame"]

        best_row = search_result["best_row"]
        validation_rows.append(
            {
                "model_key": model_name,
                "model_name": MODEL_LABELS[model_name],
                "params": json.dumps(best_params, sort_keys=True),
                "feature_profile": search_result["feature_profile"],
                "feature_count": search_result["feature_count"],
                "search_mode": search_result["search_mode"],
                "search_candidates": search_result["search_candidates"],
                "validation_auc": best_row["validation_auc"],
                "validation_ks": best_row["validation_ks"],
                "validation_brier": best_row["validation_brier"],
                "validation_12m_auc": best_row["validation_12m_auc"],
                "validation_12m_brier": best_row["validation_12m_brier"],
                "calibration_summary": best_row["calibration_summary"],
                "validation_metrics": best_row["validation_metrics"],
                "validation_metrics_12m": best_row["validation_metrics_12m"],
            }
        )

    validation_table = rank_validation_results(pd.DataFrame(validation_rows))
    validation_table["validation_rank"] = range(1, len(validation_table) + 1)
    champion_row = validation_table.iloc[0].to_dict()
    champion_key = champion_row["model_key"]

    train_validation_panel = pd.concat(
        [prepared["train"]["df"], prepared["validation"]["df"]],
        ignore_index=True,
    )
    final_models = {}
    test_predictions = {}
    final_test_rows = []
    for model_name in config.get("enabled_models", []):
        feature_profile = _hazard_feature_profile_for_model(model_name, "calendar_time")
        final_payload = _build_calendar_hazard_panel_payload(
            train_validation_panel,
            feature_profile=feature_profile,
            meta_columns=prepared["meta_columns"],
            mlp_sampling_config=config.get("mlp_sampling", {}),
            model_name=model_name,
        )
        final_model = _fit_hazard_model_from_payload(
            model_name,
            best_params_by_model[model_name],
            final_payload,
            prepared["meta_columns"],
            feature_mode="calendar_time",
        )
        final_model.update(
            {
                "modeling_type": PD_MODELING_TYPE_CALENDAR_TIME_HAZARD,
                "hazard_impute_values": prepared["hazard_impute_values"],
                "data_cutoff": prepared["data_cutoff"],
                "prepared_data": prepared,
                "mlp_sampling": config.get("mlp_sampling", {}),
            }
        )
        final_models[model_name] = final_model

        test_prediction = _score_calendar_time_hazard_model_bundle(
            final_model,
            prepared["test"]["df"],
        )
        test_metrics = _safe_evaluate_model(
            test_prediction[TARGET_1M_COLUMN],
            test_prediction["predicted_hazard_1m"],
        )
        test_metrics_12m = _safe_evaluate_model(
            test_prediction["actual_default_12m"],
            test_prediction["predicted_pd_12m"],
        )
        test_predictions[model_name] = test_prediction

        validation_rank = int(
            validation_table.loc[
                validation_table["model_key"] == model_name,
                "validation_rank",
            ].iloc[0]
        )
        final_test_rows.append(
            {
                "model_key": model_name,
                "model_name": MODEL_LABELS[model_name],
                "validation_rank": validation_rank,
                "params": json.dumps(best_params_by_model[model_name], sort_keys=True),
                "feature_profile": final_model["feature_profile"],
                "feature_count": len(final_model["feature_columns"]),
                "search_mode": "refit_on_train_plus_validation",
                "test_auc": test_metrics["AUC"],
                "test_ks": test_metrics["KS"],
                "test_brier": test_metrics["Brier"],
                "test_12m_auc": test_metrics_12m["AUC"],
                "test_12m_brier": test_metrics_12m["Brier"],
                "calibration_summary": test_metrics["calibration_summary"],
                "test_metrics": test_metrics,
                "test_metrics_12m": test_metrics_12m,
            }
        )

    final_test_table = (
        pd.DataFrame(final_test_rows)
        .sort_values("validation_rank")
        .reset_index(drop=True)
    )
    champion_test_row = final_test_table.loc[
        final_test_table["model_key"] == champion_key
    ].iloc[0].to_dict()

    champion_period_table = _build_calendar_champion_period_table(
        champion_key,
        prepared,
        train_only_models[champion_key],
        final_models[champion_key],
    )
    importance_tables = _build_importance_tables(
        final_models=final_models,
        feature_columns=prepared["feature_columns"],
    )
    combined_grid_search_table = _combine_search_tables(grid_search_tables)

    return {
        "validation_table": validation_table,
        "final_test_table": final_test_table,
        "champion_validation": champion_row,
        "champion_test": champion_test_row,
        "champion_period_table": champion_period_table,
        "champion_predictions": test_predictions[champion_key],
        "validation_predictions_by_model": validation_predictions,
        "test_predictions_by_model": test_predictions,
        "train_only_models": train_only_models,
        "final_models": final_models,
        "best_params_by_model": best_params_by_model,
        "grid_search_tables": grid_search_tables,
        "grid_search_table": combined_grid_search_table,
        "prepared_data": prepared,
        "importance_tables": importance_tables,
        "selected_model_key": champion_key,
    }


def run_manual_sweep(config, df, model_name):
    prepared = prepare_experiment_data(config, df)
    if prepared.get("modeling_type") == PD_MODELING_TYPE_HAZARD:
        return _run_hazard_manual_sweep(config, prepared, model_name)

    train_payload = prepared["train"]
    validation_payload = prepared["validation"]
    sweep_candidates = config.get("manual_sweep", {}).get(model_name, [])
    rows = []

    for index, param_override in enumerate(sweep_candidates, start=1):
        candidate_params = copy.deepcopy(config.get("model_params", {}).get(model_name, {}))
        candidate_params.update(param_override)

        model = _build_model(model_name, candidate_params)
        model.fit(train_payload["X"], train_payload["y"])
        validation_pred = model.predict_proba(validation_payload["X"])[:, 1]
        validation_metrics = evaluate_model(validation_payload["y"], validation_pred)
        rows.append(
            {
                "sweep_id": index,
                "model_key": model_name,
                "model_name": MODEL_LABELS[model_name],
                "parameter_override": json.dumps(param_override, sort_keys=True),
                "validation_auc": validation_metrics["AUC"],
                "validation_ks": validation_metrics["KS"],
                "validation_brier": validation_metrics["Brier"],
                "calibration_summary": validation_metrics["calibration_summary"],
            }
        )

    return rank_validation_results(pd.DataFrame(rows))


def run_feature_search(config, df):
    feature_search_config = copy.deepcopy(config.get("feature_search", {}))
    candidate_flags = feature_search_config.get(
        "candidate_flags",
        list(FEATURE_FLAG_CANDIDATES.keys()),
    )
    search_model = feature_search_config.get(
        "search_model",
        config.get("enabled_models", ["xgboost"])[0],
    )

    baseline_config = copy.deepcopy(config)
    baseline_config["enabled_models"] = [search_model]
    for flag in candidate_flags:
        baseline_config.setdefault("feature_flags", {})[flag] = False

    baseline_result = _evaluate_feature_flag_bundle(
        baseline_config,
        df,
        search_model,
    )
    baseline_metrics = baseline_result["best_row"]
    baseline_feature_columns = baseline_result["prepared_data"]["feature_columns"]

    rows = []
    for flag_name in candidate_flags:
        candidate_config = copy.deepcopy(baseline_config)
        candidate_config.setdefault("feature_flags", {})[flag_name] = True
        candidate_result = _evaluate_feature_flag_bundle(
            candidate_config,
            df,
            search_model,
        )

        candidate_metrics = candidate_result["best_row"]
        candidate_feature_columns = candidate_result["prepared_data"]["feature_columns"]
        added_columns = [
            column
            for column in candidate_feature_columns
            if column not in baseline_feature_columns
        ]
        rows.append(
            {
                "feature_flag": flag_name,
                "feature_group": FEATURE_FLAG_CANDIDATES.get(flag_name, flag_name),
                "search_model": MODEL_LABELS[search_model],
                "search_mode": candidate_result["search_mode"],
                "params": json.dumps(candidate_result["best_params"], sort_keys=True),
                "feature_count": len(candidate_feature_columns),
                "added_feature_count": len(added_columns),
                "added_columns_preview": ", ".join(added_columns[:8]),
                "validation_auc": candidate_metrics["validation_auc"],
                "validation_ks": candidate_metrics["validation_ks"],
                "validation_brier": candidate_metrics["validation_brier"],
                "delta_auc": candidate_metrics["validation_auc"] - baseline_metrics["validation_auc"],
                "delta_ks": candidate_metrics["validation_ks"] - baseline_metrics["validation_ks"],
                "delta_brier": candidate_metrics["validation_brier"] - baseline_metrics["validation_brier"],
                "calibration_summary": candidate_metrics["calibration_summary"],
            }
        )

    feature_search_table = rank_feature_search_results(pd.DataFrame(rows))
    if not feature_search_table.empty:
        feature_search_table["feature_rank"] = range(1, len(feature_search_table) + 1)

    return {
        "baseline_metrics": baseline_metrics,
        "baseline_feature_columns": baseline_feature_columns,
        "search_model_key": search_model,
        "search_model_name": MODEL_LABELS[search_model],
        "feature_search_table": feature_search_table,
    }


def fit_locked_model(config, df, model_name=None):
    try:
        prepared = prepare_experiment_data(config, df)
    except (KeyError, ValueError):
        if config.get("modeling_type") in {
            PD_MODELING_TYPE_HAZARD,
            PD_MODELING_TYPE_CALENDAR_TIME_HAZARD,
        }:
            raise
        prepared = _prepare_static_experiment_data(config, df)

    selected_model = model_name or config.get("selected_model_key")
    if selected_model is None:
        enabled_models = config.get("enabled_models", [])
        if len(enabled_models) != 1:
            raise ValueError(
                "fit_locked_model requires selected_model_key or exactly one enabled model."
            )
        selected_model = enabled_models[0]

    params = copy.deepcopy(config.get("model_params", {}).get(selected_model, {}))
    if prepared.get("modeling_type") == PD_MODELING_TYPE_CALENDAR_TIME_HAZARD:
        train_validation_panel = pd.concat(
            [prepared["train"]["df"], prepared["validation"]["df"]],
            ignore_index=True,
        )
        feature_profile = _hazard_feature_profile_for_model(selected_model, "calendar_time")
        panel_payload = _build_calendar_hazard_panel_payload(
            train_validation_panel,
            feature_profile=feature_profile,
            meta_columns=prepared["meta_columns"],
            mlp_sampling_config=config.get("mlp_sampling", {}),
            model_name=selected_model,
        )
        model_bundle = _fit_hazard_model_from_payload(
            selected_model,
            params,
            panel_payload,
            prepared["meta_columns"],
            feature_mode="calendar_time",
        )
        model_bundle.update(
            {
                "modeling_type": PD_MODELING_TYPE_CALENDAR_TIME_HAZARD,
                "params": params,
                "model_params": params,
                "hazard_feature_mode": "calendar_time",
                "hazard_impute_values": prepared["hazard_impute_values"],
                "meta_columns": prepared["meta_columns"],
                "data_cutoff": prepared.get("data_cutoff"),
                "prepared_data": prepared,
                "mlp_sampling": config.get("mlp_sampling", {}),
                "hazard_calibration": copy.deepcopy(config.get("hazard_calibration", {})),
                "stage2_scoring_chunk_rows": int(
                    config.get("stage2_scoring_chunk_rows", 4000)
                ),
            }
        )
        return model_bundle

    if prepared.get("modeling_type") == PD_MODELING_TYPE_HAZARD:
        train_validation_loans = pd.concat(
            [prepared["train"]["df"], prepared["validation"]["df"]],
            ignore_index=True,
        )
        model_bundle = _fit_hazard_model_bundle(
            selected_model,
            params,
            train_validation_loans,
            prepared["meta_columns"],
            feature_mode=prepared.get("hazard_feature_mode", "loan_month"),
        )
        model_bundle.update(
            {
                "params": params,
                "model_params": params,
                "hazard_feature_mode": prepared.get("hazard_feature_mode", "loan_month"),
                "hazard_impute_values": prepared["hazard_impute_values"],
                "meta_columns": prepared["meta_columns"],
                "prepared_data": prepared,
            }
        )
        return model_bundle

    train_validation_payload = combine_training_frames(
        prepared["train"],
        prepared["validation"],
    )
    model = _build_model(selected_model, params)
    model.fit(train_validation_payload["X"], train_validation_payload["y"])

    return {
        "model_key": selected_model,
        "model_name": MODEL_LABELS[selected_model],
        "params": params,
        "model": model,
        "feature_columns": prepared["feature_columns"],
        "meta_columns": prepared["meta_columns"],
        "prepared_data": prepared,
    }


def score_trained_model(
    model_bundle,
    feature_df,
    output_column="predicted_pd",
    start_month_column="months_on_book",
):
    if model_bundle.get("modeling_type") == PD_MODELING_TYPE_CALENDAR_TIME_HAZARD:
        loan_frame = prepare_pd_hazard_loan_frame(
            feature_df,
            impute_values=model_bundle.get("hazard_impute_values"),
            require_outcomes=False,
        )
        snapshot_month = model_bundle.get("data_cutoff", DEFAULT_CALENDAR_DATA_CUTOFF)
        if SNAPSHOT_MONTH_COLUMN in feature_df.columns:
            snapshot_lookup = pd.Series(
                pd.to_datetime(feature_df[SNAPSHOT_MONTH_COLUMN], errors="coerce")
                .dt.to_period("M")
                .dt.to_timestamp()
                .to_numpy(),
                index=feature_df[ID_COLUMN].to_numpy(),
            )
            loan_frame[SNAPSHOT_MONTH_COLUMN] = loan_frame[ID_COLUMN].map(snapshot_lookup)
        else:
            loan_frame[SNAPSHOT_MONTH_COLUMN] = pd.Timestamp(snapshot_month).to_period("M").to_timestamp()
        loan_frame[SNAPSHOT_MONTH_COLUMN] = loan_frame[SNAPSHOT_MONTH_COLUMN].fillna(
            pd.Timestamp(snapshot_month).to_period("M").to_timestamp()
        )
        loan_frame[TARGET_1M_COLUMN] = 0
        loan_frame[SPLIT_LABEL_COLUMN] = "scoring"
        scoring_panel = _add_calendar_time_features(loan_frame)
        scoring_panel = _apply_hazard_imputation(
            scoring_panel,
            model_bundle.get("hazard_impute_values", {}),
        )
        return _score_calendar_time_hazard_model_bundle(
            model_bundle,
            scoring_panel,
            output_column=output_column,
        )

    if model_bundle.get("modeling_type") == PD_MODELING_TYPE_HAZARD:
        loan_frame = prepare_pd_hazard_loan_frame(
            feature_df,
            impute_values=model_bundle.get("hazard_impute_values"),
            require_outcomes=False,
        )
        prediction_frame = _score_hazard_model_bundle(
            model_bundle,
            loan_frame,
            start_month_column=start_month_column,
            output_column=output_column,
        )
        return prediction_frame

    meta_columns = [
        column
        for column in model_bundle.get("meta_columns", [])
        if column in feature_df.columns
    ]
    prediction_frame = feature_df[meta_columns].copy() if meta_columns else pd.DataFrame()
    aligned_features = _align_feature_frame(
        feature_df,
        model_bundle["feature_columns"],
    )
    prediction_frame[output_column] = model_bundle["model"].predict_proba(aligned_features)[:, 1]
    prediction_frame["model_name"] = model_bundle["model_name"]
    return prediction_frame


def rank_validation_results(results_df):
    if results_df.empty:
        return results_df.copy()

    return results_df.sort_values(
        by=["validation_auc", "validation_ks", "validation_brier"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def rank_feature_search_results(results_df):
    if results_df.empty:
        return results_df.copy()

    return results_df.sort_values(
        by=["delta_auc", "delta_ks", "delta_brier", "validation_auc"],
        ascending=[False, False, True, False],
    ).reset_index(drop=True)


def prepare_pd_hazard_loan_frame(
    df,
    train_end=None,
    valid_end=None,
    impute_values=None,
    require_outcomes=True,
):
    working = ensure_sample_id(df)
    if (
        require_outcomes
        and not {STATUS_COLUMN, "home_ownership", "purpose"}.issubset(working.columns)
        and TARGET_COLUMN in working.columns
    ):
        working = reconstruct_raw_like_dataset(working)

    prepared = working.copy()
    if ISSUE_DATE_COLUMN in prepared.columns:
        prepared[ISSUE_DATE_COLUMN] = pd.to_datetime(prepared[ISSUE_DATE_COLUMN], errors="coerce")
    elif ISSUE_RAW_COLUMN in prepared.columns:
        prepared[ISSUE_DATE_COLUMN] = pd.to_datetime(
            prepared[ISSUE_RAW_COLUMN],
            format="%b-%Y",
            errors="coerce",
        )
    else:
        raise KeyError("Hazard PD modeling requires issue_date or issue_d.")

    if STATUS_COLUMN not in prepared.columns:
        if TARGET_COLUMN in prepared.columns:
            prepared[STATUS_COLUMN] = prepared[TARGET_COLUMN].map(
                {0: "Fully Paid", 1: "Charged Off"}
            )
        elif require_outcomes:
            raise KeyError("Hazard PD modeling requires loan_status or default.")
        else:
            prepared[STATUS_COLUMN] = "Unknown"

    if require_outcomes:
        prepared = prepared[prepared[STATUS_COLUMN].isin(LOSS_STATUS_VALUES)].copy()

    _ensure_hazard_input_columns(prepared)
    prepared = _clean_hazard_numeric_columns(prepared)
    prepared = _add_hazard_engineered_features(prepared)
    prepared = _add_hazard_observation_columns(prepared)

    if require_outcomes:
        prepared["actual_default"] = (prepared[STATUS_COLUMN] == "Charged Off").astype(int)
        prepared["resolved_flag"] = prepared[STATUS_COLUMN].isin(RESOLVED_LOAN_STATUSES).astype(int)
        prepared["active_flag"] = prepared[STATUS_COLUMN].isin(ACTIVE_LOAN_STATUSES).astype(int)
        prepared["actual_default_12m"] = (
            (prepared["actual_default"] == 1) & (prepared["months_on_book"] <= 12)
        ).astype(int)
    else:
        prepared["actual_default"] = (
            prepared[STATUS_COLUMN].eq("Charged Off").astype(int)
            if STATUS_COLUMN in prepared.columns
            else 0
        )
        prepared["resolved_flag"] = prepared[STATUS_COLUMN].isin(RESOLVED_LOAN_STATUSES).astype(int)
        prepared["active_flag"] = prepared[STATUS_COLUMN].isin(ACTIVE_LOAN_STATUSES).astype(int)
        prepared["actual_default_12m"] = (
            (prepared["actual_default"] == 1) & (prepared["months_on_book"] <= 12)
        ).astype(int)
    prepared[TARGET_COLUMN] = prepared["actual_default"]

    if train_end is not None and valid_end is not None:
        prepared = assign_temporal_split_labels(
            prepared,
            date_col=ISSUE_DATE_COLUMN,
            train_end=train_end,
            valid_end=valid_end,
            output_col=SPLIT_LABEL_COLUMN,
        )
    elif SPLIT_LABEL_COLUMN not in prepared.columns:
        prepared[SPLIT_LABEL_COLUMN] = pd.NA

    prepared = prepared.dropna(subset=[ISSUE_DATE_COLUMN, "term_months"]).copy()
    prepared = prepared[prepared["term_months"].isin([36, 60])].copy()
    prepared["term_months"] = prepared["term_months"].astype(int)
    prepared["months_on_book"] = prepared["months_on_book"].clip(
        lower=0,
        upper=prepared["term_months"],
    )
    prepared[OBSERVED_MONTHS_COLUMN] = (
        prepared["months_on_book"].fillna(0).round().astype(int).clip(lower=1)
    )
    prepared[OBSERVED_MONTHS_COLUMN] = np.minimum(
        prepared[OBSERVED_MONTHS_COLUMN],
        prepared["term_months"],
    )
    prepared["remaining_term"] = (
        prepared["term_months"] - prepared["months_on_book"].fillna(0)
    ).clip(lower=0)

    for column in HAZARD_CATEGORICAL_COLUMNS:
        prepared[column] = prepared[column].astype("object").fillna("unknown")
    for column in ["grade", "annual_income_band", "purpose_group"]:
        if column in prepared.columns:
            prepared[column] = prepared[column].astype("object").fillna("unknown")

    if impute_values is not None:
        prepared = _apply_hazard_imputation(prepared, impute_values)

    return prepared.sort_values([ISSUE_DATE_COLUMN, ID_COLUMN]).reset_index(drop=True)


def _ensure_hazard_input_columns(df):
    for column in HAZARD_NUMERIC_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan
    for column in HAZARD_CATEGORICAL_COLUMNS:
        if column not in df.columns:
            df[column] = "unknown"
    if "grade" not in df.columns:
        df["grade"] = "unknown"
    if "annual_income_band" not in df.columns:
        annual_income = pd.to_numeric(df["annual_inc"], errors="coerce").fillna(0).clip(lower=0)
        df["annual_income_band"] = pd.cut(
            annual_income,
            bins=ANNUAL_INCOME_BINS,
            labels=ANNUAL_INCOME_LABELS,
            right=False,
            include_lowest=True,
        ).astype("object").fillna(ANNUAL_INCOME_LABELS[0])
    if "term" not in df.columns:
        df["term"] = df["term_months"].astype(str) + " months" if "term_months" in df.columns else ""
    if "purpose_group" not in df.columns:
        df["purpose_group"] = df["purpose"].where(
            df["purpose"].isin(["debt_consolidation", "credit_card", "home_improvement", "major_purchase"]),
            "other",
        )


def _clean_hazard_numeric_columns(df):
    cleaned = df.copy()
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

    cleaned["int_rate"] = _parse_percent_like_series(cleaned["int_rate"])
    cleaned["revol_util"] = _parse_percent_like_series(cleaned["revol_util"])
    cleaned["emp_length"] = cleaned["emp_length"].apply(parse_emp_length_value)
    cleaned["missing_emp_length_flag"] = cleaned["emp_length"].isna().astype(int)
    cleaned["term_months"] = pd.to_numeric(
        cleaned["term_months"]
        if "term_months" in cleaned.columns and cleaned["term_months"].notna().any()
        else cleaned["term"].astype(str).str.extract(r"(\d+)")[0],
        errors="coerce",
    )

    cleaned.loc[(cleaned["dti"] < 0) | (cleaned["dti"] > DTI_UPPER_BOUND), "dti"] = np.nan
    cleaned.loc[
        (cleaned["int_rate"] < 0) | (cleaned["int_rate"] > RATE_UPPER_BOUND),
        "int_rate",
    ] = np.nan
    cleaned.loc[
        (cleaned["revol_util"] < 0) | (cleaned["revol_util"] > REVOL_UTIL_UPPER_BOUND),
        "revol_util",
    ] = np.nan

    for column in [
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
    ]:
        cleaned.loc[cleaned[column] < 0, column] = np.nan

    return cleaned


def _add_hazard_engineered_features(df):
    engineered = df.copy()
    safe_income = engineered["annual_inc"].clip(lower=0)
    income_denominator = safe_income.replace(0, np.nan)

    engineered["log_annual_inc"] = np.log1p(safe_income)
    engineered["loan_to_income"] = engineered["loan_amnt"] / income_denominator
    engineered["installment_to_income"] = engineered["installment"] / income_denominator
    engineered["revol_bal_to_income"] = engineered["revol_bal"] / income_denominator
    for column in ["loan_to_income", "installment_to_income", "revol_bal_to_income"]:
        engineered[column] = engineered[column].replace([np.inf, -np.inf], np.nan)

    engineered["high_dti_flag"] = (engineered["dti"] >= HIGH_DTI_THRESHOLD).astype(int)
    engineered["long_term_flag"] = (engineered["term_months"] >= 60).astype(int)
    engineered["high_revol_util_flag"] = (
        engineered["revol_util"] >= HIGH_REVOL_UTIL_THRESHOLD
    ).astype(int)
    engineered["recent_inquiry_flag"] = (
        engineered["inq_last_6mths"] >= RECENT_INQUIRY_THRESHOLD
    ).astype(int)
    engineered["prior_delinquency_flag"] = (engineered["delinq_2yrs"] > 0).astype(int)
    engineered["bankruptcy_flag"] = (engineered["pub_rec_bankruptcies"] > 0).astype(int)

    if "fico_bucket" not in engineered.columns or engineered["fico_bucket"].isna().all():
        fico_bucket = pd.cut(
            engineered["fico_range_low"],
            bins=FICO_BUCKET_BINS,
            labels=FICO_BUCKET_LABELS,
            right=False,
        )
        engineered["fico_bucket"] = fico_bucket.astype("object").fillna("unknown")
    return engineered


def _add_hazard_observation_columns(df):
    observed = df.copy()
    if "months_on_book" in observed.columns and observed["months_on_book"].notna().any():
        observed["months_on_book"] = pd.to_numeric(observed["months_on_book"], errors="coerce")
    else:
        if LAST_PAYMENT_DATE_COLUMN in observed.columns:
            observed[LAST_PAYMENT_DATE_COLUMN] = pd.to_datetime(
                observed[LAST_PAYMENT_DATE_COLUMN],
                errors="coerce",
            )
        elif LAST_PAYMENT_RAW_COLUMN in observed.columns:
            observed[LAST_PAYMENT_DATE_COLUMN] = pd.to_datetime(
                observed[LAST_PAYMENT_RAW_COLUMN],
                format="%b-%Y",
                errors="coerce",
            )
        else:
            observed[LAST_PAYMENT_DATE_COLUMN] = pd.NaT

        observed["months_on_book"] = month_diff(
            observed[ISSUE_DATE_COLUMN],
            observed[LAST_PAYMENT_DATE_COLUMN],
        )
        missing_months = observed["months_on_book"].isna()
        observed.loc[missing_months, "months_on_book"] = observed.loc[
            missing_months,
            "term_months",
        ]

    observed["months_on_book"] = pd.to_numeric(
        observed["months_on_book"],
        errors="coerce",
    ).fillna(0)
    observed["months_on_book"] = observed["months_on_book"].clip(lower=0)
    return observed


def _fit_apply_hazard_imputation(df):
    train_df = df[df[SPLIT_LABEL_COLUMN] == "train"]
    impute_values = {
        column: float(train_df[column].median())
        if column in train_df.columns and pd.notna(train_df[column].median())
        else 0.0
        for column in HAZARD_NUMERIC_COLUMNS
    }
    return _apply_hazard_imputation(df, impute_values), impute_values


def _apply_hazard_imputation(df, impute_values):
    filled = df.copy()
    for column, value in impute_values.items():
        if column in filled.columns:
            filled[column] = pd.to_numeric(filled[column], errors="coerce").fillna(value)
    return filled


def _apply_loan_level_sampling(df, sample_frac, random_state):
    if sample_frac >= 1.0:
        return df.reset_index(drop=True)

    sampled_partitions = []
    for index, (split_name, partition) in enumerate(df.groupby(SPLIT_LABEL_COLUMN, dropna=False)):
        if partition.empty:
            continue
        sample_size = max(1, int(round(len(partition) * sample_frac)))
        if sample_size >= len(partition):
            sampled_partitions.append(partition)
            continue
        if partition["actual_default"].nunique() > 1 and sample_size > 1:
            sampled, _ = train_test_split(
                partition,
                train_size=sample_size,
                stratify=partition["actual_default"],
                random_state=random_state + index,
            )
        else:
            sampled = partition.sample(n=sample_size, random_state=random_state + index)
        sampled_partitions.append(sampled)

    return (
        pd.concat(sampled_partitions, ignore_index=True)
        .sort_values([ISSUE_DATE_COLUMN, ID_COLUMN])
        .reset_index(drop=True)
    )


def _build_calendar_survival_panel_frame(
    loan_df,
    data_cutoff=DEFAULT_CALENDAR_DATA_CUTOFF,
    train_end=DEFAULT_TEMPORAL_SPLIT["train_end"],
    valid_end=DEFAULT_TEMPORAL_SPLIT["valid_end"],
    include_scoring_snapshot=True,
):
    data_cutoff = pd.Timestamp(data_cutoff).to_period("M").to_timestamp()
    working = loan_df.copy()
    working[ISSUE_DATE_COLUMN] = pd.to_datetime(
        working[ISSUE_DATE_COLUMN],
        errors="coerce",
    ).dt.to_period("M").dt.to_timestamp()
    if LAST_PAYMENT_DATE_COLUMN not in working.columns:
        working[LAST_PAYMENT_DATE_COLUMN] = pd.NaT
    working[LAST_PAYMENT_DATE_COLUMN] = pd.to_datetime(
        working[LAST_PAYMENT_DATE_COLUMN],
        errors="coerce",
    ).dt.to_period("M").dt.to_timestamp()
    working = working.dropna(subset=[ISSUE_DATE_COLUMN, "term_months"]).copy()
    working = working[working["term_months"].isin([36, 60])].copy()

    rows = []
    event_statuses = {"Charged Off", "Default"}
    payoff_statuses = {"Fully Paid"}
    train_end = pd.Timestamp(train_end).to_period("M").to_timestamp()
    valid_end = pd.Timestamp(valid_end).to_period("M").to_timestamp()

    for row in working.itertuples(index=False):
        issue_month = getattr(row, ISSUE_DATE_COLUMN)
        term_months = int(getattr(row, "term_months"))
        maturity_month = issue_month + pd.DateOffset(months=term_months)
        last_payment_month = getattr(row, LAST_PAYMENT_DATE_COLUMN)
        status = getattr(row, STATUS_COLUMN)

        event_month = pd.NaT
        if status in event_statuses and pd.notna(last_payment_month) and last_payment_month <= data_cutoff:
            event_month = min(last_payment_month, maturity_month)

        payoff_month = pd.NaT
        if status in payoff_statuses and pd.notna(last_payment_month) and last_payment_month <= data_cutoff:
            payoff_month = min(last_payment_month, maturity_month)

        censor_month = min(
            [month for month in [maturity_month, data_cutoff, payoff_month] if pd.notna(month)]
        )
        terminal_month = event_month if pd.notna(event_month) else censor_month
        if pd.isna(terminal_month) or terminal_month < issue_month:
            continue

        max_eval_snapshot = min(terminal_month, data_cutoff) - pd.DateOffset(months=1)
        snapshot_month = issue_month
        while snapshot_month <= max_eval_snapshot:
            base = row._asdict()
            base[SNAPSHOT_MONTH_COLUMN] = snapshot_month
            base[TARGET_1M_COLUMN] = int(
                pd.notna(event_month)
                and snapshot_month == event_month - pd.DateOffset(months=1)
            )
            base[SPLIT_LABEL_COLUMN] = _calendar_split_label(
                snapshot_month,
                train_end=train_end,
                valid_end=valid_end,
                data_cutoff=data_cutoff,
            )
            rows.append(base)
            snapshot_month = snapshot_month + pd.DateOffset(months=1)

        at_risk_at_cutoff = (
            include_scoring_snapshot
            and issue_month <= data_cutoff
            and data_cutoff < min([month for month in [event_month, payoff_month, maturity_month] if pd.notna(month)])
        )
        if at_risk_at_cutoff:
            base = row._asdict()
            base[SNAPSHOT_MONTH_COLUMN] = data_cutoff
            base[TARGET_1M_COLUMN] = np.nan
            base[SPLIT_LABEL_COLUMN] = "scoring"
            rows.append(base)

    if not rows:
        return pd.DataFrame()

    panel = pd.DataFrame(rows)
    panel = _add_calendar_time_features(panel)
    panel["actual_default"] = panel[TARGET_1M_COLUMN].fillna(0).astype(int)
    panel["actual_default_12m"] = panel["target_12m"].fillna(0).astype(int)
    panel = panel.sort_values([SNAPSHOT_MONTH_COLUMN, ID_COLUMN]).reset_index(drop=True)
    return panel


def _calendar_split_label(snapshot_month, train_end, valid_end, data_cutoff):
    if snapshot_month >= data_cutoff:
        return "scoring"
    if snapshot_month < train_end:
        return "train"
    if snapshot_month < valid_end:
        return "validation"
    return "test"


def _add_calendar_time_features(df):
    enriched = df.copy()
    enriched[SNAPSHOT_MONTH_COLUMN] = pd.to_datetime(
        enriched[SNAPSHOT_MONTH_COLUMN],
        errors="coerce",
    ).dt.to_period("M").dt.to_timestamp()
    enriched[ISSUE_DATE_COLUMN] = pd.to_datetime(
        enriched[ISSUE_DATE_COLUMN],
        errors="coerce",
    ).dt.to_period("M").dt.to_timestamp()
    enriched["month_on_book"] = month_diff(
        enriched[ISSUE_DATE_COLUMN],
        enriched[SNAPSHOT_MONTH_COLUMN],
    ).fillna(0).clip(lower=0)
    enriched["term_months"] = pd.to_numeric(enriched["term_months"], errors="coerce").fillna(0)
    enriched["remaining_term"] = (enriched["term_months"] - enriched["month_on_book"]).clip(lower=0)
    enriched["calendar_year"] = enriched[SNAPSHOT_MONTH_COLUMN].dt.year
    enriched["calendar_quarter"] = (
        enriched[SNAPSHOT_MONTH_COLUMN].dt.year.astype("Int64").astype(str)
        + "Q"
        + enriched[SNAPSHOT_MONTH_COLUMN].dt.quarter.astype("Int64").astype(str)
    )
    enriched["vintage_year"] = enriched[ISSUE_DATE_COLUMN].dt.year
    enriched["vintage_quarter"] = (
        enriched[ISSUE_DATE_COLUMN].dt.year.astype("Int64").astype(str)
        + "Q"
        + enriched[ISSUE_DATE_COLUMN].dt.quarter.astype("Int64").astype(str)
    )
    enriched = _add_hazard_time_features(enriched)
    enriched["scheduled_balance_proxy"] = scheduled_balance_proxy(
        funded_amount=enriched["funded_amnt"] if "funded_amnt" in enriched.columns else enriched["loan_amnt"],
        installment=enriched["installment"],
        annual_rate=enriched["int_rate"],
        months_elapsed=enriched["month_on_book"],
    )
    funded = (
        pd.to_numeric(enriched["funded_amnt"], errors="coerce")
        if "funded_amnt" in enriched.columns
        else pd.to_numeric(enriched["loan_amnt"], errors="coerce")
    )
    enriched["scheduled_balance_to_funded"] = (
        enriched["scheduled_balance_proxy"] / funded.replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0)

    event_rows = enriched[TARGET_1M_COLUMN].fillna(0).astype(int).eq(1)
    event_snapshot_lookup = enriched.loc[event_rows, [ID_COLUMN, SNAPSHOT_MONTH_COLUMN]].rename(
        columns={SNAPSHOT_MONTH_COLUMN: "event_snapshot_month"}
    )
    enriched = enriched.merge(event_snapshot_lookup, on=ID_COLUMN, how="left")
    months_to_event = month_diff(enriched[SNAPSHOT_MONTH_COLUMN], enriched["event_snapshot_month"])
    enriched["target_12m"] = (
        enriched["event_snapshot_month"].notna()
        & months_to_event.between(0, 11, inclusive="both")
    ).astype(int)
    enriched = enriched.drop(columns=["event_snapshot_month"])
    return enriched


def _fit_apply_calendar_hazard_imputation(df):
    train_df = df[df[SPLIT_LABEL_COLUMN] == "train"]
    numeric_columns = _calendar_numeric_columns(df)
    impute_values = {
        column: float(train_df[column].median())
        if column in train_df.columns and pd.notna(train_df[column].median())
        else 0.0
        for column in numeric_columns
    }
    return _apply_hazard_imputation(df, impute_values), impute_values


def _calendar_numeric_columns(df):
    extras = [
        "calendar_year",
        "vintage_year",
        "month_on_book",
        "remaining_term",
        "seasoning_ratio",
        "scheduled_balance_proxy",
        "scheduled_balance_to_funded",
    ]
    return [
        column
        for column in dict.fromkeys(HAZARD_NUMERIC_COLUMNS + extras)
        if column in df.columns and column not in HAZARD_LEAKAGE_COLUMNS
    ]


def _apply_calendar_panel_sampling(df, sample_frac, random_state):
    if sample_frac >= 1.0 or df.empty:
        return df.reset_index(drop=True)
    sampled = []
    for index, (split_name, partition) in enumerate(df.groupby(SPLIT_LABEL_COLUMN, dropna=False)):
        if split_name == "scoring":
            sampled.append(partition)
            continue
        sample_size = max(1, int(round(len(partition) * sample_frac)))
        if sample_size >= len(partition):
            sampled.append(partition)
        else:
            sampled.append(partition.sample(n=sample_size, random_state=random_state + index))
    return pd.concat(sampled, ignore_index=True).sort_values(
        [SNAPSHOT_MONTH_COLUMN, ID_COLUMN]
    ).reset_index(drop=True)


def _build_calendar_panel_counts(panel):
    if panel.empty:
        return pd.DataFrame()
    return (
        panel.groupby([SPLIT_LABEL_COLUMN, "month_bucket"], dropna=False)
        .agg(
            exposure_count=(ID_COLUMN, "size"),
            event_count=(TARGET_1M_COLUMN, "sum"),
            mean_target_1m=(TARGET_1M_COLUMN, "mean"),
        )
        .reset_index()
    )


def _write_frame_with_parquet_fallback(df, output_path):
    output_path = str(output_path)
    try:
        df.to_parquet(output_path, index=False)
    except Exception:
        fallback_path = output_path.rsplit(".", 1)[0] + ".csv"
        df.to_csv(fallback_path, index=False)


def _calendar_panel_payload(df, meta_columns):
    y = df[TARGET_1M_COLUMN].fillna(0).astype(int)
    return {
        "modeling_type": PD_MODELING_TYPE_CALENDAR_TIME_HAZARD,
        "df": df.copy(),
        "y": y,
        "meta": df[meta_columns].copy() if meta_columns else None,
        "meta_columns": meta_columns,
    }


def _calendar_partition_summary(name, df):
    return {
        "partition": name,
        "rows": int(len(df)),
        "event_rate_1m": float(df[TARGET_1M_COLUMN].mean()),
        "min_snapshot_month": df[SNAPSHOT_MONTH_COLUMN].min(),
        "max_snapshot_month": df[SNAPSHOT_MONTH_COLUMN].max(),
        "loan_count": int(df[ID_COLUMN].nunique()) if ID_COLUMN in df.columns else int(len(df)),
    }


def _hazard_loan_payload(df, meta_columns):
    return {
        "modeling_type": PD_MODELING_TYPE_HAZARD,
        "df": df.copy(),
        "y": df["actual_default"].copy(),
        "meta": df[meta_columns].copy() if meta_columns else None,
        "meta_columns": meta_columns,
    }


def _hazard_partition_summary(name, df):
    return {
        "partition": name,
        "rows": int(len(df)),
        "default_rate": float(df["actual_default"].mean()),
        "min_issue_date": df[ISSUE_DATE_COLUMN].min(),
        "max_issue_date": df[ISSUE_DATE_COLUMN].max(),
    }


def _parse_percent_like_series(series):
    text_series = series.astype(str).str.replace("%", "", regex=False).str.strip()
    return pd.to_numeric(text_series, errors="coerce")


def _fit_hazard_model_bundle(
    model_name,
    params,
    loan_df,
    meta_columns,
    feature_mode="loan_month",
):
    panel_payload = _build_hazard_panel_payload(
        loan_df,
        feature_profile=_hazard_feature_profile_for_model(model_name, feature_mode),
        meta_columns=meta_columns,
    )
    return _fit_hazard_model_from_payload(
        model_name,
        params,
        panel_payload,
        meta_columns,
        feature_mode,
    )


def _fit_hazard_model_from_payload(
    model_name,
    params,
    panel_payload,
    meta_columns,
    feature_mode="loan_month",
):
    model = _build_model(model_name, params)
    fit_kwargs = {}
    if "sample_weight" in panel_payload and panel_payload["sample_weight"] is not None:
        if model_name == "logistic_regression":
            fit_kwargs["classifier__sample_weight"] = panel_payload["sample_weight"]
        else:
            fit_kwargs["sample_weight"] = panel_payload["sample_weight"]
    model.fit(panel_payload["X"], panel_payload["y"], **fit_kwargs)

    return {
        "modeling_type": PD_MODELING_TYPE_HAZARD,
        "model_key": model_name,
        "model_name": MODEL_LABELS[model_name],
        "params": copy.deepcopy(params),
        "model_params": copy.deepcopy(params),
        "model": model,
        "feature_profile": panel_payload["feature_profile"],
        "hazard_feature_mode": feature_mode,
        "feature_columns": panel_payload["feature_columns"],
        "meta_columns": meta_columns,
    }


def _build_hazard_panel_payload(loan_df, feature_profile, meta_columns, panel_counts=None):
    if feature_profile in {GROUPED_COMPACT_HAZARD_PROFILE, GROUPED_FULL_HAZARD_PROFILE}:
        return _build_grouped_hazard_panel_payload(
            loan_df,
            feature_profile=feature_profile,
            meta_columns=meta_columns,
            panel_counts=panel_counts,
        )

    panel = _build_hazard_panel_frame(loan_df)
    if panel.empty:
        raise ValueError("Hazard panel construction produced no training rows.")
    design = _build_hazard_design_matrix(panel, feature_profile=feature_profile)
    return {
        "modeling_type": PD_MODELING_TYPE_HAZARD,
        "feature_profile": feature_profile,
        "feature_columns": list(design.columns),
        "X": design,
        "y": panel["target"].astype(int),
        "sample_weight": pd.Series(np.ones(len(panel)), index=panel.index),
        "meta": panel[[column for column in meta_columns if column in panel.columns]].copy()
        if meta_columns
        else None,
        "df": panel,
    }


def _build_calendar_hazard_panel_payload(
    panel_df,
    feature_profile,
    meta_columns,
    mlp_sampling_config=None,
    model_name=None,
):
    panel = panel_df[panel_df[TARGET_1M_COLUMN].notna()].copy()
    if model_name == "mlp_neural_network":
        panel = _sample_calendar_panel_for_mlp(panel, mlp_sampling_config or {})
    if panel.empty:
        raise ValueError("Calendar-time hazard panel construction produced no training rows.")
    design = _build_hazard_design_matrix(panel, feature_profile=feature_profile)
    if "sample_weight" in panel.columns:
        sample_weight = pd.to_numeric(panel["sample_weight"], errors="coerce").fillna(1.0)
    else:
        sample_weight = pd.Series(np.ones(len(panel)), index=panel.index)
    if model_name == "mlp_neural_network":
        sample_weight = None
    return {
        "modeling_type": PD_MODELING_TYPE_CALENDAR_TIME_HAZARD,
        "feature_profile": feature_profile,
        "feature_columns": list(design.columns),
        "X": design,
        "y": panel[TARGET_1M_COLUMN].astype(int),
        "sample_weight": sample_weight,
        "meta": panel[[column for column in meta_columns if column in panel.columns]].copy()
        if meta_columns
        else None,
        "df": panel,
    }


def _sample_calendar_panel_for_mlp(panel, sampling_config):
    positive = panel[panel[TARGET_1M_COLUMN].astype(int) == 1]
    negative = panel[panel[TARGET_1M_COLUMN].astype(int) == 0]
    ratio = int(sampling_config.get("negative_to_positive_ratio", 20))
    max_rows = int(sampling_config.get("max_train_rows", 2_000_000))
    random_state = int(sampling_config.get("random_state", 42))
    if positive.empty:
        sampled = negative.sample(
            n=min(len(negative), max_rows),
            random_state=random_state,
        )
    else:
        negative_n = min(len(negative), max_rows - len(positive), len(positive) * ratio)
        negative_n = max(0, negative_n)
        sampled_negative = (
            negative.sample(n=negative_n, random_state=random_state)
            if negative_n < len(negative)
            else negative
        )
        sampled = pd.concat([positive, sampled_negative], ignore_index=True)
        if len(sampled) > max_rows:
            sampled = sampled.sample(n=max_rows, random_state=random_state)
    return sampled.sort_values([SNAPSHOT_MONTH_COLUMN, ID_COLUMN]).reset_index(drop=True)


def _build_grouped_hazard_panel_payload(
    loan_df,
    feature_profile,
    meta_columns,
    panel_counts=None,
):
    if panel_counts is None:
        panel_counts = _build_grouped_hazard_panel_counts(loan_df)
    model_frame = _expand_panel_counts_for_model(panel_counts)
    if model_frame.empty:
        raise ValueError("Grouped hazard panel construction produced no training rows.")
    design = _build_hazard_design_matrix(model_frame, feature_profile=feature_profile)
    return {
        "modeling_type": PD_MODELING_TYPE_HAZARD,
        "feature_profile": feature_profile,
        "feature_columns": list(design.columns),
        "X": design,
        "y": model_frame["target"].astype(int),
        "sample_weight": model_frame["sample_weight"].astype(float),
        "meta": None,
        "df": model_frame,
    }


def _build_grouped_hazard_panel_counts(loan_df):
    grouped_input = loan_df.copy()
    grouped_input["event_default"] = grouped_input["actual_default"].astype(int)
    grouped_input["observed_months"] = grouped_input[OBSERVED_MONTHS_COLUMN].astype(int)
    return build_monthly_hazard_panel_counts(grouped_input)


def _expand_panel_counts_for_model(panel_counts):
    positive = panel_counts[panel_counts["event_count"] > 0].copy()
    positive["target"] = 1
    positive["sample_weight"] = positive["event_count"]

    negative = panel_counts[
        (panel_counts["exposure_count"] - panel_counts["event_count"]) > 0
    ].copy()
    negative["target"] = 0
    negative["sample_weight"] = negative["exposure_count"] - negative["event_count"]
    return pd.concat([positive, negative], ignore_index=True)


def _build_hazard_panel_frame(loan_df, chunk_size=100000):
    panel_chunks = []
    working = loan_df.copy().reset_index(drop=True)
    for start in range(0, len(working), chunk_size):
        chunk = working.iloc[start : start + chunk_size].copy()
        durations = chunk[OBSERVED_MONTHS_COLUMN].fillna(0).round().astype(int).clip(lower=0)
        valid_mask = durations > 0
        if not valid_mask.any():
            continue
        chunk = chunk.loc[valid_mask].copy()
        durations = durations.loc[valid_mask].to_numpy(dtype=int)

        repeated_index = np.repeat(np.arange(len(chunk)), durations)
        repeated = chunk.iloc[repeated_index].reset_index(drop=True)
        duration_offsets = np.repeat(np.cumsum(durations) - durations, durations)
        month_on_book = np.arange(durations.sum()) - duration_offsets + 1
        repeated["month_on_book"] = month_on_book
        repeated["remaining_term"] = (
            repeated["term_months"].astype(int) - repeated["month_on_book"]
        ).clip(lower=0)
        repeated["target"] = (
            (repeated["actual_default"].to_numpy(dtype=int) == 1)
            & (repeated["month_on_book"].to_numpy(dtype=int) == np.repeat(durations, durations))
        ).astype(int)
        repeated = _add_hazard_time_features(repeated)
        panel_chunks.append(repeated)

    if not panel_chunks:
        return pd.DataFrame()
    return pd.concat(panel_chunks, ignore_index=True)


def _score_calendar_time_hazard_model_bundle(
    model_bundle,
    panel_df,
    output_column="predicted_pd",
):
    scoring = panel_df.copy().reset_index(drop=True)
    if scoring.empty:
        return scoring.assign(
            predicted_hazard_1m=pd.Series(dtype=float),
            predicted_pd_12m=pd.Series(dtype=float),
            **{output_column: pd.Series(dtype=float)},
        )

    design = _build_hazard_design_matrix(
        scoring,
        feature_profile=model_bundle["feature_profile"],
        design_columns=model_bundle["feature_columns"],
    )
    scoring["predicted_hazard_1m"] = _apply_hazard_probability_calibration(
        model_bundle["model"].predict_proba(design)[:, 1],
        model_bundle,
    )

    score_frames = []
    chunk_rows = int(model_bundle.get("stage2_scoring_chunk_rows", 4000) or 4000)
    row_ids = np.arange(len(scoring))
    for start in range(0, len(scoring), chunk_rows):
        end = min(start + chunk_rows, len(scoring))
        future_rows = _build_calendar_future_hazard_rows_vectorized(
            scoring.iloc[start:end],
            row_ids[start:end],
        )
        if future_rows.empty:
            continue
        future_design = _build_hazard_design_matrix(
            future_rows,
            feature_profile=model_bundle["feature_profile"],
            design_columns=model_bundle["feature_columns"],
        )
        future_rows["hazard_prob"] = _apply_hazard_probability_calibration(
            model_bundle["model"].predict_proba(future_design)[:, 1],
            model_bundle,
        )
        score_frames.append(
            _aggregate_calendar_hazard_probabilities(future_rows, output_column)
        )

    if score_frames:
        pd_scores = pd.concat(score_frames, ignore_index=True)
    else:
        pd_scores = pd.DataFrame(
            columns=["_score_row_id", output_column, "predicted_pd_12m"]
        )

    scoring["_score_row_id"] = np.arange(len(scoring))
    prediction_frame = scoring.merge(pd_scores, on="_score_row_id", how="left")
    prediction_frame[output_column] = prediction_frame[output_column].fillna(0).clip(0, 1)
    prediction_frame["predicted_pd_12m"] = (
        prediction_frame["predicted_pd_12m"].fillna(0).clip(0, 1)
    )
    prediction_frame["actual_default"] = prediction_frame[TARGET_1M_COLUMN].fillna(0).astype(int)
    prediction_frame["actual_default_12m"] = (
        prediction_frame.get("target_12m", prediction_frame["actual_default"])
        .fillna(0)
        .astype(int)
    )
    prediction_frame["hazard_model_key"] = model_bundle["model_key"]
    prediction_frame["hazard_model_name"] = model_bundle["model_name"]
    prediction_frame["model_name"] = model_bundle["model_name"]
    return prediction_frame.drop(columns=["_score_row_id"], errors="ignore")


def _apply_hazard_probability_calibration(probabilities, model_bundle):
    calibrated = pd.Series(probabilities).astype(float).clip(0.0, 1.0).to_numpy()
    calibration = model_bundle.get("hazard_calibration") or {}
    if calibration.get("method") != "logit_intercept":
        return calibrated.clip(0.0, 1.0)

    shift = float(calibration.get("intercept_shift", 0.0))
    eps = float(calibration.get("epsilon", 1e-6))
    clipped = np.clip(calibrated, eps, 1.0 - eps)
    logits = np.log(clipped / (1.0 - clipped)) + shift
    logits = np.clip(logits, -50.0, 50.0)
    return (1.0 / (1.0 + np.exp(-logits))).clip(0.0, 1.0)


def _build_calendar_future_hazard_rows_vectorized(scoring_df, row_ids):
    remaining = (
        pd.to_numeric(scoring_df["remaining_term"], errors="coerce")
        .fillna(0)
        .round()
        .clip(lower=0)
        .astype(int)
        .to_numpy()
    )
    valid_mask = remaining > 0
    if not valid_mask.any():
        return pd.DataFrame()

    base = scoring_df.loc[valid_mask].reset_index(drop=True)
    base_row_ids = np.asarray(row_ids)[valid_mask]
    counts = remaining[valid_mask]
    repeated_positions = np.repeat(np.arange(len(base)), counts)
    horizons = np.concatenate([np.arange(1, count + 1, dtype=np.int16) for count in counts])

    future = base.iloc[repeated_positions].copy().reset_index(drop=True)
    future["_score_row_id"] = np.repeat(base_row_ids, counts)
    future["horizon_month"] = horizons

    snapshot = pd.to_datetime(base[SNAPSHOT_MONTH_COLUMN], errors="coerce")
    base_total_month = (
        snapshot.dt.year.fillna(1970).astype(int).to_numpy() * 12
        + snapshot.dt.month.fillna(1).astype(int).to_numpy()
        - 1
    )
    future_total_month = np.repeat(base_total_month, counts) + horizons - 1
    future[SNAPSHOT_MONTH_COLUMN] = pd.to_datetime(
        {
            "year": future_total_month // 12,
            "month": future_total_month % 12 + 1,
            "day": np.ones(len(future_total_month), dtype=np.int8),
        }
    )

    base_month_on_book = (
        pd.to_numeric(base["month_on_book"], errors="coerce").fillna(0).to_numpy()
    )
    term_months = pd.to_numeric(base["term_months"], errors="coerce").fillna(0).to_numpy()
    future["month_on_book"] = np.repeat(base_month_on_book, counts) + horizons - 1
    future["remaining_term"] = np.maximum(
        np.repeat(term_months, counts) - future["month_on_book"],
        0,
    )
    future[TARGET_1M_COLUMN] = 0
    return _add_calendar_time_features(future)


def _build_calendar_future_hazard_rows(scoring_df):
    rows = []
    for row_id, row in enumerate(scoring_df.itertuples(index=False)):
        remaining_term = int(max(0, round(getattr(row, "remaining_term", 0))))
        if remaining_term <= 0:
            continue
        horizons = np.arange(1, remaining_term + 1)
        base = {column: getattr(row, column) for column in scoring_df.columns}
        frame = pd.DataFrame({column: [value] * len(horizons) for column, value in base.items()})
        frame["_score_row_id"] = row_id
        frame["horizon_month"] = horizons
        frame[SNAPSHOT_MONTH_COLUMN] = pd.to_datetime(base[SNAPSHOT_MONTH_COLUMN]) + pd.to_timedelta(
            (horizons - 1) * 31,
            unit="D",
        )
        frame[SNAPSHOT_MONTH_COLUMN] = frame[SNAPSHOT_MONTH_COLUMN].dt.to_period("M").dt.to_timestamp()
        frame["month_on_book"] = pd.to_numeric(base["month_on_book"], errors="coerce") + horizons - 1
        frame["remaining_term"] = np.maximum(pd.to_numeric(base["term_months"], errors="coerce") - frame["month_on_book"], 0)
        rows.append(_add_calendar_time_features(frame))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _aggregate_calendar_hazard_probabilities(future_rows, output_column):
    future = future_rows.copy()
    hazard = pd.to_numeric(future["hazard_prob"], errors="coerce").fillna(0).clip(0, 1)
    future["log_survival_factor"] = np.log1p(-hazard)
    lifetime = (
        future.groupby("_score_row_id", as_index=False)["log_survival_factor"]
        .sum()
        .rename(columns={"log_survival_factor": "log_survival_lifetime"})
    )
    within_12 = future[future["horizon_month"] <= 12]
    pd_12m = (
        within_12.groupby("_score_row_id", as_index=False)["log_survival_factor"]
        .sum()
        .rename(columns={"log_survival_factor": "log_survival_12m"})
    )
    scores = lifetime.merge(pd_12m, on="_score_row_id", how="left")
    scores["log_survival_12m"] = scores["log_survival_12m"].fillna(0.0)
    scores[output_column] = (1.0 - np.exp(scores["log_survival_lifetime"])).clip(0, 1)
    scores["predicted_pd_12m"] = (
        1.0 - np.exp(scores["log_survival_12m"])
    ).clip(0, 1)
    return scores[["_score_row_id", output_column, "predicted_pd_12m"]]


def _score_hazard_model_bundle(
    model_bundle,
    loan_df,
    start_month_column="months_on_book",
    output_column="predicted_pd",
):
    scoring = loan_df.copy().reset_index(drop=True)
    if scoring.empty:
        return scoring.assign(
            **{
                output_column: pd.Series(dtype=float),
                "predicted_pd_12m": pd.Series(dtype=float),
                "model_name": model_bundle["model_name"],
            }
        )

    if start_month_column is None or start_month_column not in scoring.columns:
        scoring["start_month"] = 0
    else:
        scoring["start_month"] = (
            scoring[start_month_column].fillna(0).round().astype(int).clip(lower=0)
        )
    scoring["term_months"] = scoring["term_months"].round().astype(int).clip(lower=1)
    scoring["start_month"] = np.minimum(scoring["start_month"], scoring["term_months"])

    if model_bundle["feature_profile"] in {
        GROUPED_COMPACT_HAZARD_PROFILE,
        GROUPED_FULL_HAZARD_PROFILE,
    }:
        return _score_grouped_hazard_model_bundle(
            model_bundle,
            scoring,
            output_column=output_column,
        )

    future_rows = _build_future_hazard_rows(scoring)
    if future_rows.empty:
        pd_scores = pd.DataFrame(
            {
                ID_COLUMN: scoring[ID_COLUMN],
                output_column: 0.0,
                "predicted_pd_12m": 0.0,
            }
        )
    else:
        design_matrix = _build_hazard_design_matrix(
            future_rows,
            feature_profile=model_bundle["feature_profile"],
            design_columns=model_bundle["feature_columns"],
        )
        future_rows["hazard_prob"] = (
            model_bundle["model"].predict_proba(design_matrix)[:, 1].clip(0, 1)
        )
        pd_scores = _aggregate_hazard_probabilities(future_rows, output_column)

    meta_columns = [
        column
        for column in model_bundle.get("meta_columns", [])
        if column in scoring.columns
    ]
    prediction_frame = scoring[meta_columns].copy() if meta_columns else pd.DataFrame()
    if ID_COLUMN not in prediction_frame.columns and ID_COLUMN in scoring.columns:
        prediction_frame[ID_COLUMN] = scoring[ID_COLUMN]
    prediction_frame = prediction_frame.merge(pd_scores, on=ID_COLUMN, how="left")
    prediction_frame[output_column] = prediction_frame[output_column].fillna(0).clip(0, 1)
    prediction_frame["predicted_pd_12m"] = (
        prediction_frame["predicted_pd_12m"].fillna(0).clip(0, 1)
    )
    prediction_frame["actual_default"] = scoring["actual_default"].to_numpy()
    prediction_frame["actual_default_12m"] = scoring["actual_default_12m"].to_numpy()
    prediction_frame["hazard_model_key"] = model_bundle["model_key"]
    prediction_frame["hazard_model_name"] = model_bundle["model_name"]
    prediction_frame["model_name"] = model_bundle["model_name"]
    if "months_on_book" in scoring.columns and "months_on_book" not in prediction_frame.columns:
        prediction_frame["months_on_book"] = scoring["months_on_book"].to_numpy()
    if "remaining_term" in scoring.columns and "remaining_term" not in prediction_frame.columns:
        prediction_frame["remaining_term"] = scoring["remaining_term"].to_numpy()
    if ID_COLUMN in prediction_frame.columns:
        prediction_frame = prediction_frame.sort_values(ID_COLUMN).reset_index(drop=True)
    return prediction_frame


def _score_grouped_hazard_model_bundle(model_bundle, scoring, output_column):
    scoring = scoring.copy().reset_index(drop=True)
    scoring["_score_row_id"] = np.arange(len(scoring))
    scoring["term_months"] = scoring["term_months"].round().astype(int).clip(lower=1)
    scoring["start_month"] = (
        scoring["start_month"].fillna(0).round().astype(int).clip(lower=0)
    )
    scoring["start_month"] = np.minimum(scoring["start_month"], scoring["term_months"])
    for column in GROUPED_HAZARD_STATE_COLUMNS:
        if column not in scoring.columns:
            scoring[column] = "unknown"
    scoring["term_months"] = scoring["term_months"].astype(int)
    for column in [c for c in GROUPED_HAZARD_STATE_COLUMNS if c != "term_months"]:
        scoring[column] = scoring[column].astype("object").fillna("unknown")

    base_states = scoring[GROUPED_HAZARD_STATE_COLUMNS].drop_duplicates().copy()
    hazard_state_rows = []
    for term_months, state_frame in base_states.groupby("term_months", dropna=False):
        term_months = int(term_months)
        if term_months <= 0:
            continue
        month_values = np.arange(1, term_months + 1)
        repeated = state_frame.loc[state_frame.index.repeat(term_months)].reset_index(
            drop=True
        )
        repeated["month_on_book"] = np.tile(month_values, len(state_frame))
        repeated["remaining_term"] = np.maximum(
            term_months - repeated["month_on_book"],
            0,
        )
        hazard_state_rows.append(_add_hazard_time_features(repeated))

    if not hazard_state_rows:
        pd_scores = scoring[[ID_COLUMN]].copy()
        pd_scores[output_column] = 0.0
        pd_scores["predicted_pd_12m"] = 0.0
    else:
        hazard_frame = pd.concat(hazard_state_rows, ignore_index=True)
        design_matrix = _build_hazard_design_matrix(
            hazard_frame,
            feature_profile=model_bundle["feature_profile"],
            design_columns=model_bundle["feature_columns"],
        )
        hazard_frame["hazard_prob"] = (
            model_bundle["model"].predict_proba(design_matrix)[:, 1].clip(0, 1)
        )
        hazard_frame["log_survival_factor"] = np.log1p(
            -hazard_frame["hazard_prob"].clip(0, 1 - 1e-7)
        )
        hazard_frame = hazard_frame.sort_values(
            GROUPED_HAZARD_STATE_COLUMNS + ["month_on_book"]
        )
        hazard_frame["cum_log_survival"] = hazard_frame.groupby(
            GROUPED_HAZARD_STATE_COLUMNS,
            dropna=False,
        )["log_survival_factor"].cumsum()

        zero_rows = base_states.copy()
        zero_rows["month_on_book"] = 0
        zero_rows["cum_log_survival"] = 0.0
        cumulative_lookup = pd.concat(
            [
                hazard_frame[
                    GROUPED_HAZARD_STATE_COLUMNS
                    + ["month_on_book", "cum_log_survival"]
                ],
                zero_rows[
                    GROUPED_HAZARD_STATE_COLUMNS
                    + ["month_on_book", "cum_log_survival"]
                ],
            ],
            ignore_index=True,
        )

        score_rows = scoring[
            ["_score_row_id", ID_COLUMN] + GROUPED_HAZARD_STATE_COLUMNS + ["start_month"]
        ].copy()
        score_rows["life_end_month"] = score_rows["term_months"]
        score_rows["end_12m_month"] = np.minimum(
            score_rows["start_month"] + 12,
            score_rows["term_months"],
        )
        score_rows = _merge_cumulative_hazard_lookup(
            score_rows,
            cumulative_lookup,
            "start_month",
            "cum_log_start",
        )
        score_rows = _merge_cumulative_hazard_lookup(
            score_rows,
            cumulative_lookup,
            "life_end_month",
            "cum_log_lifetime_end",
        )
        score_rows = _merge_cumulative_hazard_lookup(
            score_rows,
            cumulative_lookup,
            "end_12m_month",
            "cum_log_12m_end",
        )
        for column in ["cum_log_start", "cum_log_lifetime_end", "cum_log_12m_end"]:
            score_rows[column] = score_rows[column].fillna(0.0)
        score_rows[output_column] = (
            1.0
            - np.exp(score_rows["cum_log_lifetime_end"] - score_rows["cum_log_start"])
        ).clip(0, 1)
        score_rows["predicted_pd_12m"] = (
            1.0 - np.exp(score_rows["cum_log_12m_end"] - score_rows["cum_log_start"])
        ).clip(0, 1)
        pd_scores = score_rows[[ID_COLUMN, output_column, "predicted_pd_12m"]].copy()

    meta_columns = [
        column
        for column in model_bundle.get("meta_columns", [])
        if column in scoring.columns
    ]
    prediction_frame = scoring[meta_columns].copy() if meta_columns else pd.DataFrame()
    if ID_COLUMN not in prediction_frame.columns and ID_COLUMN in scoring.columns:
        prediction_frame[ID_COLUMN] = scoring[ID_COLUMN]
    prediction_frame = prediction_frame.merge(pd_scores, on=ID_COLUMN, how="left")
    prediction_frame[output_column] = prediction_frame[output_column].fillna(0).clip(0, 1)
    prediction_frame["predicted_pd_12m"] = (
        prediction_frame["predicted_pd_12m"].fillna(0).clip(0, 1)
    )
    prediction_frame["actual_default"] = scoring["actual_default"].to_numpy()
    prediction_frame["actual_default_12m"] = scoring["actual_default_12m"].to_numpy()
    prediction_frame["hazard_model_key"] = model_bundle["model_key"]
    prediction_frame["hazard_model_name"] = model_bundle["model_name"]
    prediction_frame["model_name"] = model_bundle["model_name"]
    if "months_on_book" in scoring.columns and "months_on_book" not in prediction_frame.columns:
        prediction_frame["months_on_book"] = scoring["months_on_book"].to_numpy()
    if "remaining_term" in scoring.columns and "remaining_term" not in prediction_frame.columns:
        prediction_frame["remaining_term"] = scoring["remaining_term"].to_numpy()
    if ID_COLUMN in prediction_frame.columns:
        prediction_frame = prediction_frame.sort_values(ID_COLUMN).reset_index(drop=True)
    return prediction_frame


def _merge_cumulative_hazard_lookup(score_rows, cumulative_lookup, month_column, value_column):
    lookup = cumulative_lookup.rename(
        columns={
            "month_on_book": month_column,
            "cum_log_survival": value_column,
        }
    )
    return score_rows.merge(
        lookup,
        on=GROUPED_HAZARD_STATE_COLUMNS + [month_column],
        how="left",
    )


def _build_future_hazard_rows(scoring_df):
    rows = []
    for row in scoring_df.itertuples(index=False):
        start_month = int(getattr(row, "start_month"))
        term_months = int(getattr(row, "term_months"))
        if start_month >= term_months:
            continue
        future_months = np.arange(start_month + 1, term_months + 1)
        base = {column: getattr(row, column) for column in scoring_df.columns}
        frame = pd.DataFrame({column: [value] * len(future_months) for column, value in base.items()})
        frame["month_on_book"] = future_months
        frame["remaining_term"] = np.maximum(term_months - future_months, 0)
        frame["horizon_month"] = future_months - start_month
        rows.append(_add_hazard_time_features(frame))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _aggregate_hazard_probabilities(future_rows, output_column):
    future = future_rows.copy()
    future["survival_factor"] = 1.0 - future["hazard_prob"].clip(0, 1)
    lifetime = (
        future.groupby(ID_COLUMN, as_index=False)["survival_factor"]
        .prod()
        .rename(columns={"survival_factor": "survival_lifetime"})
    )
    within_12 = future[future["horizon_month"] <= 12]
    pd_12m = (
        within_12.groupby(ID_COLUMN, as_index=False)["survival_factor"]
        .prod()
        .rename(columns={"survival_factor": "survival_12m"})
    )
    scores = lifetime.merge(pd_12m, on=ID_COLUMN, how="left")
    scores["survival_12m"] = scores["survival_12m"].fillna(1.0)
    scores[output_column] = (1.0 - scores["survival_lifetime"]).clip(0, 1)
    scores["predicted_pd_12m"] = (1.0 - scores["survival_12m"]).clip(0, 1)
    return scores[[ID_COLUMN, output_column, "predicted_pd_12m"]]


def _add_hazard_time_features(df):
    enriched = df.copy()
    enriched["month_bucket"] = _month_bucket_label(enriched["month_on_book"])
    enriched["seasoning_ratio"] = (
        enriched["month_on_book"] / enriched["term_months"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0)
    enriched["term_month_interaction"] = (
        enriched["term_months"].astype(int).astype(str)
        + "_"
        + enriched["month_bucket"].astype(str)
    )
    enriched["purpose_group_month_bucket"] = (
        enriched["purpose_group"].astype(str) + "_" + enriched["month_bucket"].astype(str)
    )
    enriched["fico_bucket_month_bucket"] = (
        enriched["fico_bucket"].astype(str) + "_" + enriched["month_bucket"].astype(str)
    )
    return enriched


def _build_hazard_design_matrix(df, feature_profile, design_columns=None):
    if feature_profile in {GROUPED_COMPACT_HAZARD_PROFILE, GROUPED_FULL_HAZARD_PROFILE}:
        categorical_columns = [
            "grade",
            "fico_bucket",
            "purpose_group",
            "annual_income_band",
            "term_months",
            "month_bucket",
        ]
        if feature_profile == GROUPED_FULL_HAZARD_PROFILE:
            categorical_columns.append("term_month_interaction")
        categorical_columns = [column for column in categorical_columns if column in df.columns]
        feature_frame = pd.get_dummies(
            df[categorical_columns].astype("object").fillna("unknown"),
            columns=categorical_columns,
            drop_first=True,
            dtype=float,
        )
        feature_frame = _sanitize_feature_frame_columns(feature_frame)
        if design_columns is None:
            return feature_frame
        aligned = feature_frame.copy()
        for column in design_columns:
            if column not in aligned.columns:
                aligned[column] = 0.0
        return aligned.reindex(columns=design_columns, fill_value=0.0)

    feature_frame = pd.DataFrame(index=df.index)
    is_calendar_profile = feature_profile in {
        COMPACT_CALENDAR_HAZARD_PROFILE,
        FULL_CALENDAR_HAZARD_PROFILE,
    }
    numeric_columns = [
        column
        for column in HAZARD_NUMERIC_COLUMNS
        + [
            "month_on_book",
            "remaining_term",
            "seasoning_ratio",
            "calendar_year",
            "vintage_year",
            "scheduled_balance_proxy",
            "scheduled_balance_to_funded",
        ]
        if column in df.columns and column not in HAZARD_LEAKAGE_COLUMNS
    ]
    for column in numeric_columns:
        feature_frame[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    categorical_columns = ["fico_bucket", "verification_status", "home_ownership", "month_bucket"]
    if is_calendar_profile:
        categorical_columns.extend(["calendar_quarter", "vintage_quarter"])
    if feature_profile in {FULL_HAZARD_PROFILE, FULL_CALENDAR_HAZARD_PROFILE}:
        categorical_columns.extend(
            [
                "purpose",
                "term_month_interaction",
                "purpose_group_month_bucket",
                "fico_bucket_month_bucket",
            ]
        )

    categorical_columns = [column for column in categorical_columns if column in df.columns]
    if categorical_columns:
        categorical_frame = pd.get_dummies(
            df[categorical_columns].astype("object").fillna("unknown"),
            columns=categorical_columns,
            drop_first=True,
            dtype=float,
        )
        feature_frame = pd.concat([feature_frame, categorical_frame], axis=1)

    if feature_profile in {COMPACT_HAZARD_PROFILE, COMPACT_CALENDAR_HAZARD_PROFILE}:
        feature_frame = feature_frame[
            [
                column
                for column in feature_frame.columns
                if not column.startswith("purpose_")
                and not column.startswith("term_month_interaction_")
                and not column.startswith("purpose_group_month_bucket_")
                and not column.startswith("fico_bucket_month_bucket_")
                and not column.startswith("grade_")
                and not column.startswith("initial_list_status_")
                and not column.startswith("application_type_")
            ]
        ]

    feature_frame = feature_frame.loc[:, ~feature_frame.columns.duplicated()].copy()
    feature_frame = _sanitize_feature_frame_columns(feature_frame)
    if design_columns is None:
        return feature_frame

    missing_columns = [column for column in design_columns if column not in feature_frame.columns]
    if missing_columns:
        missing_frame = pd.DataFrame(
            0.0,
            index=feature_frame.index,
            columns=missing_columns,
        )
        aligned = pd.concat([feature_frame, missing_frame], axis=1)
    else:
        aligned = feature_frame.copy()
    return aligned.reindex(columns=design_columns, fill_value=0.0)


def _sanitize_feature_frame_columns(feature_frame):
    sanitized_columns = []
    seen = {}
    for column in feature_frame.columns:
        sanitized = (
            str(column)
            .replace("[", "_")
            .replace("]", "_")
            .replace("<", "lt_")
            .replace(">", "gt_")
        )
        if sanitized in seen:
            seen[sanitized] += 1
            sanitized = f"{sanitized}__{seen[sanitized]}"
        else:
            seen[sanitized] = 0
        sanitized_columns.append(sanitized)
    sanitized_frame = feature_frame.copy()
    sanitized_frame.columns = sanitized_columns
    return sanitized_frame


def _hazard_feature_profile_for_model(model_name, feature_mode="loan_month"):
    if feature_mode == "calendar_time":
        if model_name == "logistic_regression":
            return COMPACT_CALENDAR_HAZARD_PROFILE
        return FULL_CALENDAR_HAZARD_PROFILE
    if feature_mode == "grouped":
        if model_name == "logistic_regression":
            return GROUPED_COMPACT_HAZARD_PROFILE
        return GROUPED_FULL_HAZARD_PROFILE
    if model_name == "logistic_regression":
        return COMPACT_HAZARD_PROFILE
    return FULL_HAZARD_PROFILE


def _month_bucket_label(month_values):
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
            HAZARD_MONTH_BUCKET_LABELS[:-1],
            default=HAZARD_MONTH_BUCKET_LABELS[-1],
        ),
        index=month_array.index,
    )


def _build_hazard_champion_period_table(
    champion_key,
    prepared,
    train_only_model,
    final_model,
):
    train_prediction = _score_hazard_model_bundle(
        train_only_model,
        prepared["train"]["df"],
        start_month_column=None,
    )
    validation_prediction = _score_hazard_model_bundle(
        train_only_model,
        prepared["validation"]["df"],
        start_month_column=None,
    )
    test_prediction = _score_hazard_model_bundle(
        final_model,
        prepared["test"]["df"],
        start_month_column=None,
    )
    return pd.DataFrame(
        [
            _hazard_period_row("train", "train only", prepared["train"]["df"], train_prediction),
            _hazard_period_row(
                "validation",
                "train only",
                prepared["validation"]["df"],
                validation_prediction,
            ),
            _hazard_period_row("test", "train + validation", prepared["test"]["df"], test_prediction),
        ]
    ).assign(model_name=MODEL_LABELS[champion_key])


def _build_calendar_champion_period_table(
    champion_key,
    prepared,
    train_only_model,
    final_model,
):
    train_prediction = _score_calendar_time_hazard_model_bundle(
        train_only_model,
        prepared["train"]["df"],
    )
    validation_prediction = _score_calendar_time_hazard_model_bundle(
        train_only_model,
        prepared["validation"]["df"],
    )
    test_prediction = _score_calendar_time_hazard_model_bundle(
        final_model,
        prepared["test"]["df"],
    )
    return pd.DataFrame(
        [
            _calendar_period_row("train", "train only", prepared["train"]["df"], train_prediction),
            _calendar_period_row(
                "validation",
                "train only",
                prepared["validation"]["df"],
                validation_prediction,
            ),
            _calendar_period_row("test", "train + validation", prepared["test"]["df"], test_prediction),
        ]
    ).assign(model_name=MODEL_LABELS[champion_key])


def _calendar_period_row(period_name, fitted_on, partition_df, prediction_df):
    metrics = _safe_evaluate_model(
        prediction_df[TARGET_1M_COLUMN],
        prediction_df["predicted_hazard_1m"],
    )
    return {
        "period": period_name,
        "fitted_on": fitted_on,
        "rows": int(len(partition_df)),
        "default_rate": float(partition_df[TARGET_1M_COLUMN].mean()),
        "min_snapshot_month": partition_df[SNAPSHOT_MONTH_COLUMN].min(),
        "max_snapshot_month": partition_df[SNAPSHOT_MONTH_COLUMN].max(),
        "auc": metrics["AUC"],
        "ks": metrics["KS"],
        "brier": metrics["Brier"],
    }


def _hazard_period_row(period_name, fitted_on, partition_df, prediction_df):
    metrics = evaluate_model(
        prediction_df["actual_default"],
        prediction_df["predicted_pd"],
    )
    return {
        "period": period_name,
        "fitted_on": fitted_on,
        "rows": int(len(partition_df)),
        "default_rate": float(partition_df["actual_default"].mean()),
        "min_issue_date": partition_df[ISSUE_DATE_COLUMN].min(),
        "max_issue_date": partition_df[ISSUE_DATE_COLUMN].max(),
        "auc": metrics["AUC"],
        "ks": metrics["KS"],
        "brier": metrics["Brier"],
    }


def _run_hazard_manual_sweep(config, prepared, model_name):
    sweep_candidates = config.get("manual_sweep", {}).get(model_name, [])
    rows = []
    for index, param_override in enumerate(sweep_candidates, start=1):
        candidate_params = copy.deepcopy(config.get("model_params", {}).get(model_name, {}))
        candidate_params.update(param_override)
        model_bundle = _fit_hazard_model_bundle(
            model_name,
            candidate_params,
            prepared["train"]["df"],
            prepared["meta_columns"],
            feature_mode=prepared.get("hazard_feature_mode", "loan_month"),
        )
        validation_prediction = _score_hazard_model_bundle(
            model_bundle,
            prepared["validation"]["df"],
            start_month_column=None,
        )
        validation_metrics = evaluate_model(
            validation_prediction["actual_default"],
            validation_prediction["predicted_pd"],
        )
        rows.append(
            {
                "sweep_id": index,
                "model_key": model_name,
                "model_name": MODEL_LABELS[model_name],
                "feature_profile": model_bundle["feature_profile"],
                "feature_count": len(model_bundle["feature_columns"]),
                "parameter_override": json.dumps(param_override, sort_keys=True),
                "validation_auc": validation_metrics["AUC"],
                "validation_ks": validation_metrics["KS"],
                "validation_brier": validation_metrics["Brier"],
                "calibration_summary": validation_metrics["calibration_summary"],
            }
        )
    return rank_validation_results(pd.DataFrame(rows))


def _search_model(model_name, config, train_payload, validation_payload):
    if train_payload.get("modeling_type") == PD_MODELING_TYPE_CALENDAR_TIME_HAZARD:
        return _search_calendar_time_hazard_model(
            model_name,
            config,
            train_payload,
            validation_payload,
        )
    if train_payload.get("modeling_type") == PD_MODELING_TYPE_HAZARD:
        return _search_hazard_model(model_name, config, train_payload, validation_payload)

    parameter_candidates, search_mode = _parameter_candidates(model_name, config)
    rows = []
    best_key = None
    best_model = None
    best_params = None
    best_metrics = None
    best_prediction_frame = None

    for candidate_id, params in enumerate(parameter_candidates, start=1):
        model = _build_model(model_name, params)
        model.fit(train_payload["X"], train_payload["y"])
        validation_pred = model.predict_proba(validation_payload["X"])[:, 1]
        validation_metrics = evaluate_model(validation_payload["y"], validation_pred)
        comparison_key = _metric_comparison_key(validation_metrics)

        rows.append(
            {
                "candidate_id": candidate_id,
                "model_key": model_name,
                "model_name": MODEL_LABELS[model_name],
                "search_mode": search_mode,
                "params": json.dumps(params, sort_keys=True),
                "params_payload": copy.deepcopy(params),
                "validation_auc": validation_metrics["AUC"],
                "validation_ks": validation_metrics["KS"],
                "validation_brier": validation_metrics["Brier"],
                "calibration_summary": validation_metrics["calibration_summary"],
                "validation_metrics": validation_metrics,
            }
        )

        if best_key is None or comparison_key > best_key:
            best_key = comparison_key
            best_model = model
            best_params = copy.deepcopy(params)
            best_metrics = validation_metrics
            best_prediction_frame = _build_prediction_frame(
                validation_payload["meta"],
                validation_payload["y"],
                validation_pred,
                model_name,
            )

    search_table = rank_validation_results(pd.DataFrame(rows))
    if not search_table.empty:
        search_table["search_rank"] = range(1, len(search_table) + 1)

    best_row = {
        "validation_auc": best_metrics["AUC"],
        "validation_ks": best_metrics["KS"],
        "validation_brier": best_metrics["Brier"],
        "calibration_summary": best_metrics["calibration_summary"],
        "validation_metrics": best_metrics,
    }

    return {
        "best_model": best_model,
        "best_params": best_params,
        "best_row": best_row,
        "prediction_frame": best_prediction_frame,
        "search_table": search_table,
        "search_mode": search_mode,
        "search_candidates": len(parameter_candidates),
    }


def _search_hazard_model(model_name, config, train_payload, validation_payload):
    parameter_candidates, search_mode = _parameter_candidates(model_name, config)
    rows = []
    best_key = None
    best_model = None
    best_params = None
    best_metrics = None
    best_metrics_12m = None
    best_prediction_frame = None
    feature_mode = config.get("hazard_feature_mode", "loan_month")
    feature_profile = _hazard_feature_profile_for_model(model_name, feature_mode)
    panel_payload = train_payload.get("hazard_panel_payloads_by_profile", {}).get(
        feature_profile
    )
    if panel_payload is None:
        panel_payload = _build_hazard_panel_payload(
            train_payload["df"],
            feature_profile=feature_profile,
            meta_columns=train_payload["meta_columns"],
        )

    for candidate_id, params in enumerate(parameter_candidates, start=1):
        model_bundle = _fit_hazard_model_from_payload(
            model_name,
            params,
            panel_payload,
            train_payload["meta_columns"],
            feature_mode=feature_mode,
        )
        validation_prediction = _score_hazard_model_bundle(
            model_bundle,
            validation_payload["df"],
            start_month_column=None,
        )
        validation_metrics = evaluate_model(
            validation_prediction["actual_default"],
            validation_prediction["predicted_pd"],
        )
        validation_metrics_12m = evaluate_model(
            validation_prediction["actual_default_12m"],
            validation_prediction["predicted_pd_12m"],
        )
        comparison_key = _metric_comparison_key(validation_metrics)

        rows.append(
            {
                "candidate_id": candidate_id,
                "model_key": model_name,
                "model_name": MODEL_LABELS[model_name],
                "feature_profile": model_bundle["feature_profile"],
                "feature_count": len(model_bundle["feature_columns"]),
                "search_mode": search_mode,
                "params": json.dumps(params, sort_keys=True),
                "params_payload": copy.deepcopy(params),
                "validation_auc": validation_metrics["AUC"],
                "validation_ks": validation_metrics["KS"],
                "validation_brier": validation_metrics["Brier"],
                "validation_12m_auc": validation_metrics_12m["AUC"],
                "validation_12m_brier": validation_metrics_12m["Brier"],
                "calibration_summary": validation_metrics["calibration_summary"],
                "validation_metrics": validation_metrics,
                "validation_metrics_12m": validation_metrics_12m,
            }
        )

        if best_key is None or comparison_key > best_key:
            best_key = comparison_key
            best_model = model_bundle
            best_params = copy.deepcopy(params)
            best_metrics = validation_metrics
            best_metrics_12m = validation_metrics_12m
            best_prediction_frame = validation_prediction

    search_table = rank_validation_results(pd.DataFrame(rows))
    if not search_table.empty:
        search_table["search_rank"] = range(1, len(search_table) + 1)

    best_row = {
        "validation_auc": best_metrics["AUC"],
        "validation_ks": best_metrics["KS"],
        "validation_brier": best_metrics["Brier"],
        "validation_12m_auc": best_metrics_12m["AUC"],
        "validation_12m_brier": best_metrics_12m["Brier"],
        "calibration_summary": best_metrics["calibration_summary"],
        "validation_metrics": best_metrics,
        "validation_metrics_12m": best_metrics_12m,
    }

    return {
        "best_model": best_model,
        "best_params": best_params,
        "best_row": best_row,
        "prediction_frame": best_prediction_frame,
        "search_table": search_table,
        "search_mode": search_mode,
        "search_candidates": len(parameter_candidates),
        "feature_profile": best_model["feature_profile"],
        "feature_count": len(best_model["feature_columns"]),
    }


def _search_calendar_time_hazard_model(model_name, config, train_payload, validation_payload):
    parameter_candidates, search_mode = _parameter_candidates(model_name, config)
    rows = []
    best_key = None
    best_model = None
    best_params = None
    best_metrics = None
    best_metrics_12m = None
    best_prediction_frame = None
    feature_profile = _hazard_feature_profile_for_model(model_name, "calendar_time")
    panel_payload = _build_calendar_hazard_panel_payload(
        train_payload["df"],
        feature_profile=feature_profile,
        meta_columns=train_payload["meta_columns"],
        mlp_sampling_config=config.get("mlp_sampling", {}),
        model_name=model_name,
    )

    for candidate_id, params in enumerate(parameter_candidates, start=1):
        model_bundle = _fit_hazard_model_from_payload(
            model_name,
            params,
            panel_payload,
            train_payload["meta_columns"],
            feature_mode="calendar_time",
        )
        model_bundle["modeling_type"] = PD_MODELING_TYPE_CALENDAR_TIME_HAZARD
        validation_prediction = _score_calendar_time_hazard_model_bundle(
            model_bundle,
            validation_payload["df"],
        )
        validation_metrics = _safe_evaluate_model(
            validation_prediction[TARGET_1M_COLUMN],
            validation_prediction["predicted_hazard_1m"],
        )
        validation_metrics_12m = _safe_evaluate_model(
            validation_prediction["actual_default_12m"],
            validation_prediction["predicted_pd_12m"],
        )
        comparison_key = _metric_comparison_key(validation_metrics)

        rows.append(
            {
                "candidate_id": candidate_id,
                "model_key": model_name,
                "model_name": MODEL_LABELS[model_name],
                "feature_profile": model_bundle["feature_profile"],
                "feature_count": len(model_bundle["feature_columns"]),
                "search_mode": search_mode,
                "params": json.dumps(params, sort_keys=True),
                "params_payload": copy.deepcopy(params),
                "validation_auc": validation_metrics["AUC"],
                "validation_ks": validation_metrics["KS"],
                "validation_brier": validation_metrics["Brier"],
                "validation_12m_auc": validation_metrics_12m["AUC"],
                "validation_12m_brier": validation_metrics_12m["Brier"],
                "calibration_summary": validation_metrics["calibration_summary"],
                "validation_metrics": validation_metrics,
                "validation_metrics_12m": validation_metrics_12m,
            }
        )

        if best_key is None or comparison_key > best_key:
            best_key = comparison_key
            best_model = model_bundle
            best_params = copy.deepcopy(params)
            best_metrics = validation_metrics
            best_metrics_12m = validation_metrics_12m
            best_prediction_frame = validation_prediction

    search_table = rank_validation_results(pd.DataFrame(rows))
    if not search_table.empty:
        search_table["search_rank"] = range(1, len(search_table) + 1)

    best_row = {
        "validation_auc": best_metrics["AUC"],
        "validation_ks": best_metrics["KS"],
        "validation_brier": best_metrics["Brier"],
        "validation_12m_auc": best_metrics_12m["AUC"],
        "validation_12m_brier": best_metrics_12m["Brier"],
        "calibration_summary": best_metrics["calibration_summary"],
        "validation_metrics": best_metrics,
        "validation_metrics_12m": best_metrics_12m,
    }

    return {
        "best_model": best_model,
        "best_params": best_params,
        "best_row": best_row,
        "prediction_frame": best_prediction_frame,
        "search_table": search_table,
        "search_mode": search_mode,
        "search_candidates": len(parameter_candidates),
        "feature_profile": best_model["feature_profile"],
        "feature_count": len(best_model["feature_columns"]),
    }


def _parameter_candidates(model_name, config):
    base_params = copy.deepcopy(config.get("model_params", {}).get(model_name, {}))
    use_grid_search = bool(config.get("use_grid_search", False))
    grid_search_models = set(
        config.get("grid_search_models", config.get("enabled_models", []))
    )
    if not use_grid_search or model_name not in grid_search_models:
        return [base_params], "fixed"

    grid_spec = copy.deepcopy(config.get("param_grid", {}).get(model_name, {}))
    if not grid_spec and config.get("use_reduced_default_grid", True):
        grid_spec = copy.deepcopy(DEFAULT_REDUCED_PARAM_GRID.get(model_name, {}))
    if config.get("reduce_grid_search", True):
        grid_spec = _reduced_grid_spec(model_name, grid_spec)
    if not grid_spec:
        return [base_params], "fixed"

    max_candidates = grid_spec.pop("max_candidates", None)
    parameter_candidates = []
    for override in ParameterGrid(grid_spec):
        candidate_params = copy.deepcopy(base_params)
        candidate_params.update(override)
        parameter_candidates.append(candidate_params)

    if not parameter_candidates:
        return [base_params], "fixed"
    if max_candidates is not None:
        parameter_candidates = _prune_parameter_candidates(
            model_name,
            parameter_candidates,
            max_candidates=int(max_candidates),
            base_params=base_params,
        )
    return parameter_candidates, "grid_search"


def _reduced_grid_spec(model_name, grid_spec):
    if not grid_spec:
        return grid_spec
    reduced = copy.deepcopy(grid_spec)
    max_candidates = reduced.get("max_candidates")
    if model_name == "logistic_regression":
        if "C" in reduced:
            reduced["C"] = [
                value
                for value in reduced["C"]
                if float(value) in {0.25, 0.5, 1.0, 2.0}
            ] or DEFAULT_REDUCED_PARAM_GRID[model_name]["C"]
    elif model_name == "random_forest":
        if "n_estimators" in reduced:
            reduced["n_estimators"] = [
                value for value in reduced["n_estimators"] if int(value) in {120, 180}
            ] or DEFAULT_REDUCED_PARAM_GRID[model_name]["n_estimators"]
    elif model_name == "hist_gradient_boosting":
        if "learning_rate" in reduced:
            reduced["learning_rate"] = [
                value for value in reduced["learning_rate"] if float(value) >= 0.03
            ] or DEFAULT_REDUCED_PARAM_GRID[model_name]["learning_rate"]
    elif model_name == "xgboost":
        max_candidates = max_candidates or DEFAULT_REDUCED_PARAM_GRID[model_name]["max_candidates"]
    elif model_name == "mlp_neural_network":
        if "hidden_layer_sizes" in reduced:
            allowed = {(32,), (64,), (64, 32)}
            reduced["hidden_layer_sizes"] = [
                tuple(value) if isinstance(value, list) else value
                for value in reduced["hidden_layer_sizes"]
            ]
            reduced["hidden_layer_sizes"] = [
                value for value in reduced["hidden_layer_sizes"] if tuple(value) in allowed
            ] or DEFAULT_REDUCED_PARAM_GRID[model_name]["hidden_layer_sizes"]
    if max_candidates is not None:
        reduced["max_candidates"] = max_candidates
    return reduced


def _prune_parameter_candidates(model_name, candidates, max_candidates, base_params):
    if len(candidates) <= max_candidates:
        return candidates

    required = []
    for required_params in [
        base_params,
        HISTORIC_STRONG_PARAM_CANDIDATES.get(model_name, {}),
    ]:
        if not required_params:
            continue
        match = _find_matching_candidate(candidates, required_params)
        if match is not None and match not in required:
            required.append(match)

    remaining = [candidate for candidate in candidates if candidate not in required]
    remaining = sorted(
        remaining,
        key=lambda params: _candidate_priority(model_name, params),
        reverse=True,
    )
    selected = required + remaining[: max(0, max_candidates - len(required))]
    selected_keys = {_params_key(params) for params in selected}
    return [candidate for candidate in candidates if _params_key(candidate) in selected_keys]


def _find_matching_candidate(candidates, required_params):
    for candidate in candidates:
        if all(candidate.get(key) == value for key, value in required_params.items()):
            return candidate
    return None


def _candidate_priority(model_name, params):
    if model_name == "xgboost":
        return (
            float(params.get("n_estimators", 0)),
            float(params.get("learning_rate", 0)),
            -abs(float(params.get("max_depth", 0)) - 6.0),
            -float(params.get("max_depth", 0)),
        )
    return tuple(float(value) for value in params.values() if isinstance(value, (int, float)))


def _params_key(params):
    return tuple(sorted((key, _hashable_param_value(value)) for key, value in params.items()))


def _hashable_param_value(value):
    if isinstance(value, list):
        return tuple(_hashable_param_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_hashable_param_value(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((key, _hashable_param_value(item)) for key, item in value.items()))
    return value


def _evaluate_feature_flag_bundle(config, df, model_name):
    prepared = prepare_experiment_data(config, df)
    search_result = _search_model(
        model_name=model_name,
        config=config,
        train_payload=prepared["train"],
        validation_payload=prepared["validation"],
    )
    return {
        "prepared_data": prepared,
        "best_row": search_result["best_row"],
        "best_params": search_result["best_params"],
        "search_mode": search_result["search_mode"],
    }


def _combine_search_tables(grid_search_tables):
    combined_tables = []
    for model_name, table in grid_search_tables.items():
        if table is None or table.empty:
            continue
        combined = table.copy()
        combined["model_key"] = model_name
        combined_tables.append(combined)

    if not combined_tables:
        return pd.DataFrame()

    return pd.concat(combined_tables, ignore_index=True)


def _build_model(model_name, params):
    if model_name not in MODEL_BUILDERS:
        raise KeyError(f"Unsupported model: {model_name}")
    return MODEL_BUILDERS[model_name](params)


def _align_feature_frame(df, feature_columns):
    aligned = df.copy()
    for column in feature_columns:
        if column not in aligned.columns:
            aligned[column] = 0
    return aligned[feature_columns].copy()


def _build_prediction_frame(meta, y_true, y_pred_proba, model_name):
    prediction_frame = meta.copy() if meta is not None else pd.DataFrame()
    prediction_frame["actual_default"] = y_true.to_numpy()
    prediction_frame["predicted_pd"] = y_pred_proba
    prediction_frame["model_name"] = MODEL_LABELS[model_name]

    if ID_COLUMN in prediction_frame.columns:
        prediction_frame = prediction_frame.sort_values(ID_COLUMN).reset_index(drop=True)
    return prediction_frame


def _build_champion_period_table(
    champion_key,
    train_payload,
    validation_payload,
    test_payload,
    train_only_model,
    final_model,
):
    train_pred = train_only_model.predict_proba(train_payload["X"])[:, 1]
    validation_pred = train_only_model.predict_proba(validation_payload["X"])[:, 1]
    test_pred = final_model.predict_proba(test_payload["X"])[:, 1]

    train_metrics = evaluate_model(train_payload["y"], train_pred)
    validation_metrics = evaluate_model(validation_payload["y"], validation_pred)
    test_metrics = evaluate_model(test_payload["y"], test_pred)

    return pd.DataFrame(
        [
            _period_row(
                "train",
                "train only",
                train_payload["df"],
                train_metrics,
            ),
            _period_row(
                "validation",
                "train only",
                validation_payload["df"],
                validation_metrics,
            ),
            _period_row(
                "test",
                "train + validation",
                test_payload["df"],
                test_metrics,
            ),
        ]
    ).assign(model_name=MODEL_LABELS[champion_key])


def _period_row(period_name, fitted_on, partition_df, metrics):
    return {
        "period": period_name,
        "fitted_on": fitted_on,
        "rows": int(len(partition_df)),
        "default_rate": float(partition_df[TARGET_COLUMN].mean()),
        "min_issue_date": partition_df[ISSUE_DATE_COLUMN].min(),
        "max_issue_date": partition_df[ISSUE_DATE_COLUMN].max(),
        "auc": metrics["AUC"],
        "ks": metrics["KS"],
        "brier": metrics["Brier"],
    }


def _build_importance_tables(final_models, feature_columns, top_n=15):
    importance_tables = {}

    logistic_model = final_models.get("logistic_regression")
    if logistic_model is not None:
        estimator = logistic_model.get("model") if isinstance(logistic_model, dict) else logistic_model
        logistic_features = (
            logistic_model.get("feature_columns", feature_columns)
            if isinstance(logistic_model, dict)
            else feature_columns
        )
        coefficients = estimator.named_steps["classifier"].coef_[0]
        coefficient_table = pd.DataFrame(
            {
                "feature": logistic_features,
                "coefficient": coefficients,
                "abs_coefficient": abs(coefficients),
            }
        )
        importance_tables["logistic_regression"] = coefficient_table.sort_values(
            "abs_coefficient",
            ascending=False,
        ).head(top_n).reset_index(drop=True)

    xgboost_model = final_models.get("xgboost")
    if xgboost_model is not None:
        estimator = xgboost_model.get("model") if isinstance(xgboost_model, dict) else xgboost_model
        xgboost_features = (
            xgboost_model.get("feature_columns", feature_columns)
            if isinstance(xgboost_model, dict)
            else feature_columns
        )
        importance_table = pd.DataFrame(
            {
                "feature": xgboost_features,
                "importance": estimator.feature_importances_,
            }
        )
        importance_tables["xgboost"] = importance_table.sort_values(
            "importance",
            ascending=False,
        ).head(top_n).reset_index(drop=True)

    return importance_tables


def _safe_evaluate_model(y_true, y_pred_proba, n_bins=10):
    y_true = pd.Series(y_true).fillna(0).astype(int)
    y_pred_proba = pd.Series(y_pred_proba).fillna(0).clip(0, 1)
    if y_true.nunique() < 2:
        return {
            "AUC": 0.5,
            "KS": 0.0,
            "Brier": float(np.mean((y_true.to_numpy() - y_pred_proba.to_numpy()) ** 2)),
            "roc_curve": {"fpr": np.array([0.0, 1.0]), "tpr": np.array([0.0, 1.0]), "thresholds": np.array([1.0, 0.0])},
            "calibration_curve": {
                "prob_true": np.array([float(y_true.mean())]),
                "prob_pred": np.array([float(y_pred_proba.mean())]),
                "bin_count": np.array([len(y_true)]),
                "bin_left": np.array([0.0]),
                "bin_right": np.array([1.0]),
            },
            "calibration_summary": "Single-class target in this split; AUC set to neutral 0.5.",
        }
    return evaluate_model(y_true, y_pred_proba, n_bins=n_bins)


def _metric_comparison_key(metrics):
    return (
        float(metrics["AUC"]),
        float(metrics["KS"]),
        -float(metrics["Brier"]),
    )
