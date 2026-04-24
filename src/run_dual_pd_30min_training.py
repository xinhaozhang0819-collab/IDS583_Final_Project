from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from calendar_bigdata import read_modeling_sample
from evaluation import evaluate_model
from experiment import (
    FULL_CALENDAR_HAZARD_PROFILE,
    ID_COLUMN,
    MODEL_LABELS,
    PD_MODELING_TYPE_CALENDAR_TIME_HAZARD,
    PD_MODELING_TYPE_STATIC,
    SNAPSHOT_MONTH_COLUMN,
    TARGET_1M_COLUMN,
    _add_calendar_time_features,
    _align_feature_frame,
    _apply_hazard_imputation,
    _build_calendar_hazard_panel_payload,
    _build_hazard_design_matrix,
    _build_model,
    _fit_hazard_model_from_payload,
    _hazard_feature_profile_for_model,
    _safe_evaluate_model,
    prepare_experiment_data,
    prepare_pd_hazard_loan_frame,
)
from loss_preprocess import (
    ACTIVE_LOAN_STATUSES,
    RESOLVED_LOAN_STATUSES,
    SPLIT_LABEL_COLUMN,
    STATUS_COLUMN,
    load_loss_raw_dataset,
    prepare_loss_workflow_dataset,
)
from preprocess import ISSUE_DATE_COLUMN, TARGET_COLUMN, build_feature_dataset_from_raw, ensure_sample_id
from run_hazard_pd_pipeline import DATA_DIR, PROCESSED_DIR, REPORT_DIR, build_loss_config
from loss_workflow import run_loss_reserve_workflow
from visualization import export_loss_reserve_visuals


RANDOM_STATE = 42
DATA_CUTOFF = pd.Timestamp("2018-12-01")
TRAIN_CAP = 300_000
VALIDATION_CAP = 50_000
TEST_CAP = 100_000
SCORING_CAP = 25_000
TRAIN_NEGATIVE_TO_POSITIVE_RATIO = 4
TRAINING_BUDGET_SECONDS = 30 * 60
SKIP_REMAINING_AFTER_SECONDS = 22 * 60
REFIT_AFTER_SECONDS = 26 * 60
FUTURE_SCORE_CHUNK_ROWS = 20_000

VIEW_PATH = PROCESSED_DIR / "calendar_survival_dual_30min_training_view.parquet"
VIEW_SUMMARY_PATH = PROCESSED_DIR / "calendar_survival_dual_30min_training_view_summary.json"
RUNTIME_PATH = PROCESSED_DIR / "dual_pd_30min_runtime.json"
SEARCH_TABLE_PATH = PROCESSED_DIR / "dual_pd_12m_hazard_search_results.csv"
STAGE2_PREDICTION_PATH = PROCESSED_DIR / "dual_pd_stage2_predictions.csv"
STAGE2_CONFIG_PATH = PROCESSED_DIR / "stage2_champion_config.json"
TEST_PD_PATH = PROCESSED_DIR / "test_with_pd_best_model.csv"
VALIDATION_12M_BACKTEST_PATH = PROCESSED_DIR / "validation_12m_observable_backtest.csv"
TEST_12M_BACKTEST_PATH = PROCESSED_DIR / "test_12m_observable_backtest.csv"
LIFETIME_DIAGNOSTIC_PATH = PROCESSED_DIR / "lifetime_resolved_diagnostic.csv"
OBSERVABLE_12M_METRICS_PATH = PROCESSED_DIR / "observable_12m_loss_metrics.csv"

STATIC_HGB_PARAMS = {
    "early_stopping": False,
    "l2_regularization": 0.0,
    "learning_rate": 0.05,
    "max_depth": 6,
    "max_iter": 150,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 50,
}
RF_12M_PARAMS = {
    "n_estimators": 90,
    "max_depth": 10,
    "min_samples_leaf": 60,
    "min_samples_split": 200,
    "max_samples": 0.7,
}
XGB_12M_CANDIDATES = [
    {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.08},
    {"n_estimators": 120, "max_depth": 4, "learning_rate": 0.06},
    {"n_estimators": 120, "max_depth": 5, "learning_rate": 0.06},
    {"n_estimators": 140, "max_depth": 4, "learning_rate": 0.05},
    {"n_estimators": 120, "max_depth": 6, "learning_rate": 0.08},
]


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _logit_shift(probabilities, shift, eps=1e-6):
    p = np.clip(np.asarray(probabilities, dtype=float), eps, 1.0 - eps)
    logits = np.log(p / (1.0 - p)) + float(shift)
    logits = np.clip(logits, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-logits))


def _fit_intercept_shift(probabilities, target_mean, weights=None):
    probabilities = np.asarray(probabilities, dtype=float)
    if weights is None:
        weights = np.ones(len(probabilities), dtype=float)
    weights = np.asarray(weights, dtype=float)
    target_mean = float(np.clip(target_mean, 1e-6, 1.0 - 1e-6))
    low, high = -50.0, 50.0
    for _ in range(100):
        mid = (low + high) / 2.0
        shifted = _logit_shift(probabilities, mid)
        shifted_mean = float(np.average(shifted, weights=weights))
        if shifted_mean < target_mean:
            low = mid
        else:
            high = mid
    return float((low + high) / 2.0)


def _static_config():
    return {
        "modeling_type": PD_MODELING_TYPE_STATIC,
        "enabled_models": ["hist_gradient_boosting"],
        "selected_model_key": "hist_gradient_boosting",
        "model_params": {"hist_gradient_boosting": copy.deepcopy(STATIC_HGB_PARAMS)},
        "random_state": RANDOM_STATE,
        "sample_frac": 1.0,
        "temporal_split": {"train_end": "2016-01-01", "valid_end": "2017-01-01"},
        "meta_columns": [
            ID_COLUMN,
            ISSUE_DATE_COLUMN,
            "loan_amnt",
            "annual_inc",
            "fico_range_low",
            "term_months",
            "int_rate",
            "revol_util",
        ],
    }


def _fit_static_lifetime_model(raw_loss_df: pd.DataFrame) -> tuple[dict, dict]:
    start = time.perf_counter()
    config = _static_config()
    prepared = prepare_experiment_data(config, raw_loss_df)
    train = prepared["train"]
    validation = prepared["validation"]
    test = prepared["test"]

    train_model = _build_model("hist_gradient_boosting", STATIC_HGB_PARAMS)
    train_model.fit(train["X"], train["y"])
    validation_raw = train_model.predict_proba(validation["X"])[:, 1]
    validation_shift = _fit_intercept_shift(validation_raw, validation["y"].mean())
    validation_calibrated = _logit_shift(validation_raw, validation_shift)
    validation_metrics_raw = evaluate_model(validation["y"], validation_raw)
    validation_metrics_calibrated = evaluate_model(validation["y"], validation_calibrated)

    train_validation_x = pd.concat([train["X"], validation["X"]], ignore_index=True)
    train_validation_y = pd.concat([train["y"], validation["y"]], ignore_index=True)
    final_model = _build_model("hist_gradient_boosting", STATIC_HGB_PARAMS)
    final_model.fit(train_validation_x, train_validation_y)
    test_raw = final_model.predict_proba(test["X"])[:, 1]
    test_calibrated = _logit_shift(test_raw, validation_shift)
    test_metrics_raw = evaluate_model(test["y"], test_raw)
    test_metrics_calibrated = evaluate_model(test["y"], test_calibrated)

    bundle = {
        "modeling_type": PD_MODELING_TYPE_STATIC,
        "model_key": "hist_gradient_boosting",
        "model_name": MODEL_LABELS["hist_gradient_boosting"],
        "params": copy.deepcopy(STATIC_HGB_PARAMS),
        "model": final_model,
        "feature_columns": prepared["feature_columns"],
        "meta_columns": prepared["meta_columns"],
        "lifetime_calibration": {
            "method": "logit_intercept",
            "intercept_shift": validation_shift,
            "target": "validation_lifetime_default_rate",
            "validation_default_rate": float(validation["y"].mean()),
            "validation_raw_mean": float(validation_raw.mean()),
            "validation_calibrated_mean": float(validation_calibrated.mean()),
        },
        "prepared_data": prepared,
    }
    summary = {
        "model_key": "hist_gradient_boosting",
        "model_name": MODEL_LABELS["hist_gradient_boosting"],
        "source": "github_original_static_hgb_rerun",
        "params": copy.deepcopy(STATIC_HGB_PARAMS),
        "train_rows": int(len(train["df"])),
        "validation_rows": int(len(validation["df"])),
        "test_rows": int(len(test["df"])),
        "validation_auc_raw": float(validation_metrics_raw["AUC"]),
        "validation_auc_calibrated": float(validation_metrics_calibrated["AUC"]),
        "validation_ks_calibrated": float(validation_metrics_calibrated["KS"]),
        "validation_brier_calibrated": float(validation_metrics_calibrated["Brier"]),
        "test_auc_raw": float(test_metrics_raw["AUC"]),
        "test_auc_calibrated": float(test_metrics_calibrated["AUC"]),
        "test_ks_calibrated": float(test_metrics_calibrated["KS"]),
        "test_brier_calibrated": float(test_metrics_calibrated["Brier"]),
        "fit_seconds": float(time.perf_counter() - start),
    }
    return bundle, summary


def _score_static_lifetime(bundle: dict, raw_df: pd.DataFrame) -> pd.DataFrame:
    feature_df = build_feature_dataset_from_raw(raw_df)
    aligned = _align_feature_frame(feature_df, bundle["feature_columns"])
    raw = bundle["model"].predict_proba(aligned)[:, 1]
    shift = bundle.get("lifetime_calibration", {}).get("intercept_shift", 0.0)
    calibrated = _logit_shift(raw, shift)
    return pd.DataFrame(
        {
            ID_COLUMN: feature_df[ID_COLUMN].astype(int).to_numpy(),
            "predicted_pd_lifetime_static_raw": raw,
            "predicted_pd_lifetime_static": calibrated,
            "lifetime_model_name": bundle["model_name"],
            "lifetime_model_source": "github_original_static_hgb_rerun",
        }
    )


def _sample_split(partition, max_rows, random_state, preserve_positive=False):
    if partition.empty or len(partition) <= max_rows:
        return partition.copy()
    positive = partition[partition[TARGET_1M_COLUMN].fillna(0).astype(int).eq(1)]
    negative = partition[partition[TARGET_1M_COLUMN].fillna(0).astype(int).eq(0)]
    if preserve_positive:
        positive_sample = positive.copy()
        negative_cap = max(0, max_rows - len(positive_sample))
        if not positive_sample.empty:
            negative_cap = min(
                negative_cap,
                len(positive_sample) * TRAIN_NEGATIVE_TO_POSITIVE_RATIO,
            )
        negative_sample = negative.sample(
            n=min(len(negative), negative_cap),
            random_state=random_state,
        )
    else:
        positive_n = int(round(max_rows * len(positive) / max(len(partition), 1)))
        positive_n = min(len(positive), max(positive_n, 1) if not positive.empty else 0)
        negative_n = min(len(negative), max_rows - positive_n)
        positive_sample = positive.sample(n=positive_n, random_state=random_state)
        negative_sample = negative.sample(n=negative_n, random_state=random_state + 1)
    return pd.concat([positive_sample, negative_sample], ignore_index=True)


def _full_panel_counts(modeling_sample: pd.DataFrame) -> dict:
    summary_path = PROCESSED_DIR / "calendar_survival_panel_summary.json"
    if summary_path.exists():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        return {
            "split_counts": {k: float(v) for k, v in payload.get("split_counts", {}).items()},
            "event_counts": {k: float(v) for k, v in payload.get("event_counts", {}).items()},
        }
    split_counts = modeling_sample["split_label"].value_counts().to_dict()
    event_counts = modeling_sample.groupby("split_label")[TARGET_1M_COLUMN].sum().to_dict()
    return {"split_counts": split_counts, "event_counts": event_counts}


def _build_training_view(modeling_sample: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    modeling_sample = modeling_sample.copy()
    full_counts = _full_panel_counts(modeling_sample)
    frames = []
    for idx, (split_name, cap) in enumerate(
        [("train", TRAIN_CAP), ("validation", VALIDATION_CAP), ("test", TEST_CAP)]
    ):
        partition = modeling_sample[modeling_sample["split_label"].eq(split_name)].copy()
        sampled = _sample_split(
            partition,
            max_rows=cap,
            random_state=RANDOM_STATE + idx * 17,
            preserve_positive=(split_name == "train"),
        )
        sampled["sample_weight"] = 1.0
        if split_name == "train" and not sampled.empty:
            y = sampled[TARGET_1M_COLUMN].fillna(0).astype(int)
            full_events = float(full_counts["event_counts"].get("train", y.sum()))
            full_rows = float(full_counts["split_counts"].get("train", len(partition)))
            sampled_events = max(float(y.sum()), 1.0)
            sampled_non_events = max(float(len(y) - y.sum()), 1.0)
            sampled["sample_weight"] = np.where(
                y.eq(1),
                full_events / sampled_events,
                max(full_rows - full_events, 1.0) / sampled_non_events,
            )
        frames.append(sampled)

    scoring = modeling_sample[modeling_sample["split_label"].eq("scoring")].copy()
    if len(scoring) > SCORING_CAP:
        scoring = scoring.sample(n=SCORING_CAP, random_state=RANDOM_STATE + 99)
    scoring["sample_weight"] = 1.0
    frames.append(scoring)
    view = pd.concat(frames, ignore_index=True)
    sort_columns = [c for c in ["split_label", SNAPSHOT_MONTH_COLUMN, ID_COLUMN] if c in view.columns]
    view = view.sort_values(sort_columns).reset_index(drop=True)
    summary = {
        "view_path": str(VIEW_PATH),
        "rows": int(len(view)),
        "split_counts": {str(k): int(v) for k, v in view["split_label"].value_counts().items()},
        "event_counts": {
            str(k): int(v)
            for k, v in view.groupby("split_label")[TARGET_1M_COLUMN].sum(min_count=1).fillna(0).items()
        },
        "sample_id_snapshot_unique": bool(
            not view.duplicated([ID_COLUMN, SNAPSHOT_MONTH_COLUMN]).any()
        ),
        "caps": {
            "train": TRAIN_CAP,
            "validation": VALIDATION_CAP,
            "test": TEST_CAP,
            "scoring": SCORING_CAP,
            "train_negative_to_positive_ratio": TRAIN_NEGATIVE_TO_POSITIVE_RATIO,
        },
        "full_panel_counts_source": str(PROCESSED_DIR / "calendar_survival_panel_summary.json"),
        "random_state": RANDOM_STATE,
    }
    return view, summary


def _hazard_config(view_path: Path) -> dict:
    return {
        "modeling_type": PD_MODELING_TYPE_CALENDAR_TIME_HAZARD,
        "hazard_feature_mode": "calendar_time",
        "calendar_panel_is_prebuilt": True,
        "calendar_modeling_sample_path": str(view_path),
        "data_cutoff": DATA_CUTOFF.strftime("%Y-%m-%d"),
        "processed_data_path": str(DATA_DIR / "raw" / "accepted_2007_to_2018Q4.csv"),
        "random_state": RANDOM_STATE,
        "sample_frac": 1.0,
        "temporal_split": {"train_end": "2016-01-01", "valid_end": "2017-01-01"},
        "enabled_models": ["logistic_regression", "random_forest", "xgboost"],
        "meta_columns": [
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
        "model_params": {
            "logistic_regression": {"C": 1.0, "max_iter": 300, "class_weight": None},
            "random_forest": {
                "n_estimators": 90,
                "max_depth": 10,
                "min_samples_split": 200,
                "min_samples_leaf": 60,
                "max_features": "sqrt",
                "bootstrap": True,
                "max_samples": 0.7,
                "n_jobs": -1,
                "random_state": RANDOM_STATE,
            },
            "xgboost": {
                "n_estimators": 120,
                "max_depth": 4,
                "learning_rate": 0.06,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 5,
                "reg_lambda": 1.0,
                "random_state": RANDOM_STATE,
                "n_jobs": max(1, min(6, (getattr(__import__("os"), "cpu_count")() or 2) // 2)),
            },
        },
        "mlp_sampling": {},
    }


def _candidate_plan(config: dict) -> list[dict]:
    xgb_base = copy.deepcopy(config["model_params"]["xgboost"])
    return [
        {
            "model_key": "logistic_regression",
            "candidate_id": 1,
            "params": copy.deepcopy(config["model_params"]["logistic_regression"]),
            "champion_eligible": False,
        },
        {
            "model_key": "random_forest",
            "candidate_id": 1,
            "params": {**copy.deepcopy(config["model_params"]["random_forest"]), **RF_12M_PARAMS},
            "champion_eligible": True,
        },
        *[
            {
                "model_key": "xgboost",
                "candidate_id": index + 1,
                "params": {**copy.deepcopy(xgb_base), **override},
                "champion_eligible": True,
            }
            for index, override in enumerate(XGB_12M_CANDIDATES)
        ],
    ]


def _fit_hazard_candidate(model_key, params, prepared, payload_cache):
    if model_key not in payload_cache:
        profile = _hazard_feature_profile_for_model(model_key, "calendar_time")
        payload_cache[model_key] = _build_calendar_hazard_panel_payload(
            prepared["train"]["df"],
            feature_profile=profile,
            meta_columns=prepared["meta_columns"],
            model_name=model_key,
        )
    bundle = _fit_hazard_model_from_payload(
        model_key,
        params,
        payload_cache[model_key],
        prepared["meta_columns"],
        feature_mode="calendar_time",
    )
    bundle.update(
        {
            "modeling_type": PD_MODELING_TYPE_CALENDAR_TIME_HAZARD,
            "hazard_impute_values": prepared["hazard_impute_values"],
            "data_cutoff": prepared["data_cutoff"],
            "prepared_data": prepared,
            "stage2_scoring_chunk_rows": FUTURE_SCORE_CHUNK_ROWS,
        }
    )
    return bundle


def _fast_hazard_score(bundle: dict, scoring_df: pd.DataFrame) -> pd.DataFrame:
    design = _build_hazard_design_matrix(
        scoring_df,
        feature_profile=bundle["feature_profile"],
        design_columns=bundle["feature_columns"],
    )
    pred = bundle["model"].predict_proba(design)[:, 1].clip(0, 1)
    return pd.DataFrame(
        {
            ID_COLUMN: scoring_df[ID_COLUMN].to_numpy(),
            "actual_default": scoring_df[TARGET_1M_COLUMN].fillna(0).astype(int).to_numpy(),
            "predicted_hazard_1m": pred,
        }
    )


def _build_future_12m_rows(scoring_df: pd.DataFrame, row_ids: np.ndarray) -> pd.DataFrame:
    remaining = (
        pd.to_numeric(scoring_df["remaining_term"], errors="coerce")
        .fillna(0)
        .round()
        .clip(lower=0, upper=12)
        .astype(int)
        .to_numpy()
    )
    valid = remaining > 0
    if not valid.any():
        return pd.DataFrame()
    base = scoring_df.loc[valid].reset_index(drop=True)
    base_row_ids = np.asarray(row_ids)[valid]
    counts = remaining[valid]
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
    term_months = pd.to_numeric(base["term_months"], errors="coerce").fillna(0).to_numpy()
    base_mob = pd.to_numeric(base["month_on_book"], errors="coerce").fillna(0).to_numpy()
    future["month_on_book"] = np.repeat(base_mob, counts) + horizons - 1
    future["remaining_term"] = np.maximum(np.repeat(term_months, counts) - future["month_on_book"], 0)
    future[TARGET_1M_COLUMN] = 0
    return _add_calendar_time_features(future)


def _prepare_hazard_scoring_panel(bundle: dict, raw_df: pd.DataFrame, snapshot_mode: str) -> pd.DataFrame:
    loan_frame = prepare_pd_hazard_loan_frame(
        raw_df,
        impute_values=bundle.get("hazard_impute_values"),
        require_outcomes=False,
    )
    if snapshot_mode == "active_cutoff":
        loan_frame[SNAPSHOT_MONTH_COLUMN] = DATA_CUTOFF
    elif snapshot_mode == "origination":
        loan_frame[SNAPSHOT_MONTH_COLUMN] = pd.to_datetime(
            loan_frame[ISSUE_DATE_COLUMN], errors="coerce"
        ).dt.to_period("M").dt.to_timestamp()
    else:
        raise ValueError(f"Unsupported snapshot_mode={snapshot_mode}")
    loan_frame[TARGET_1M_COLUMN] = 0
    loan_frame[SPLIT_LABEL_COLUMN] = "scoring"
    scoring = _add_calendar_time_features(loan_frame)
    return _apply_hazard_imputation(scoring, bundle.get("hazard_impute_values", {}))


def _score_hazard_12m(
    bundle: dict,
    raw_df: pd.DataFrame,
    snapshot_mode: str,
    calibration_shift: float | None = None,
) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame(columns=[ID_COLUMN, "predicted_hazard_1m", "predicted_pd_12m_raw", "predicted_pd_12m"])
    scoring = _prepare_hazard_scoring_panel(bundle, raw_df, snapshot_mode=snapshot_mode).reset_index(drop=True)
    current_design = _build_hazard_design_matrix(
        scoring,
        feature_profile=bundle["feature_profile"],
        design_columns=bundle["feature_columns"],
    )
    current_hazard = bundle["model"].predict_proba(current_design)[:, 1].clip(0, 1)
    score_frames = []
    row_ids = np.arange(len(scoring))
    for start in range(0, len(scoring), FUTURE_SCORE_CHUNK_ROWS):
        end = min(start + FUTURE_SCORE_CHUNK_ROWS, len(scoring))
        future_rows = _build_future_12m_rows(scoring.iloc[start:end], row_ids[start:end])
        if future_rows.empty:
            continue
        design = _build_hazard_design_matrix(
            future_rows,
            feature_profile=bundle["feature_profile"],
            design_columns=bundle["feature_columns"],
        )
        hazard = bundle["model"].predict_proba(design)[:, 1].clip(0, 1)
        future_rows["log_survival_factor"] = np.log1p(-hazard)
        scores = (
            future_rows.groupby("_score_row_id", as_index=False)["log_survival_factor"]
            .sum()
            .rename(columns={"log_survival_factor": "log_survival_12m"})
        )
        scores["predicted_pd_12m_raw"] = (1.0 - np.exp(scores["log_survival_12m"])).clip(0, 1)
        score_frames.append(scores[["_score_row_id", "predicted_pd_12m_raw"]])
    base = pd.DataFrame(
        {
            "_score_row_id": np.arange(len(scoring)),
            ID_COLUMN: scoring[ID_COLUMN].astype(int).to_numpy(),
            "predicted_hazard_1m": current_hazard,
        }
    )
    if score_frames:
        scores = pd.concat(score_frames, ignore_index=True)
        base = base.merge(scores, on="_score_row_id", how="left")
    else:
        base["predicted_pd_12m_raw"] = 0.0
    base["predicted_pd_12m_raw"] = base["predicted_pd_12m_raw"].fillna(0).clip(0, 1)
    shift = calibration_shift
    if shift is None:
        shift = bundle.get("pd_12m_calibration", {}).get("intercept_shift", 0.0)
    base["predicted_pd_12m"] = _logit_shift(base["predicted_pd_12m_raw"], shift)
    zero_remaining = scoring["remaining_term"].fillna(0).le(0).to_numpy()
    base.loc[zero_remaining, ["predicted_pd_12m_raw", "predicted_pd_12m"]] = 0.0
    return base.drop(columns=["_score_row_id"])


def _fit_12m_hazard_model(
    view: pd.DataFrame,
    workflow_df: pd.DataFrame,
    raw_loss_df: pd.DataFrame,
    start_total: float,
) -> tuple[dict, pd.DataFrame, dict]:
    config = _hazard_config(VIEW_PATH)
    prepared = prepare_experiment_data(config, view)
    candidates = _candidate_plan(config)
    payload_cache = {}
    rows = []
    bundles = {}
    skipped = []

    for index, candidate in enumerate(candidates):
        elapsed = time.perf_counter() - start_total
        low_priority_xgb = candidate["model_key"] == "xgboost" and candidate["candidate_id"] >= 4
        if elapsed > SKIP_REMAINING_AFTER_SECONDS and low_priority_xgb:
            skipped.append({**candidate, "reason": "runtime_guard_after_22_minutes"})
            continue
        if elapsed > REFIT_AFTER_SECONDS:
            skipped.append({**candidate, "reason": "runtime_guard_after_26_minutes"})
            continue
        fit_start = time.perf_counter()
        bundle = _fit_hazard_candidate(
            candidate["model_key"],
            candidate["params"],
            prepared,
            payload_cache,
        )
        fit_seconds = time.perf_counter() - fit_start
        score_start = time.perf_counter()
        validation_prediction = _fast_hazard_score(bundle, prepared["validation"]["df"])
        score_seconds = time.perf_counter() - score_start
        metrics = _safe_evaluate_model(
            validation_prediction["actual_default"],
            validation_prediction["predicted_hazard_1m"],
        )
        row = {
            "candidate_order": index + 1,
            "candidate_id": candidate["candidate_id"],
            "model_key": candidate["model_key"],
            "model_name": MODEL_LABELS[candidate["model_key"]],
            "champion_eligible": bool(candidate["champion_eligible"]),
            "params": json.dumps(candidate["params"], sort_keys=True, default=_json_default),
            "params_payload": candidate["params"],
            "feature_profile": bundle["feature_profile"],
            "feature_count": len(bundle["feature_columns"]),
            "validation_1m_auc": metrics["AUC"],
            "validation_1m_ks": metrics["KS"],
            "validation_1m_brier": metrics["Brier"],
            "validation_12m_auc": np.nan,
            "validation_12m_ks": np.nan,
            "validation_12m_brier": np.nan,
            "fit_seconds": fit_seconds,
            "score_seconds": score_seconds,
            "horizon_score_seconds": np.nan,
            "pd_12m_calibration_shift": np.nan,
        }
        rows.append(row)
        key = (candidate["model_key"], candidate["candidate_id"])
        bundles[key] = bundle
        print(
            f"12M candidate {candidate['model_key']} #{candidate['candidate_id']} "
            f"fit={fit_seconds:.1f}s score={score_seconds:.1f}s auc={metrics['AUC']:.4f}",
            flush=True,
        )

    search = pd.DataFrame(rows)
    if search.empty:
        raise RuntimeError("No 12M hazard candidates completed.")
    eligible = search[search["champion_eligible"]].copy()
    horizon_candidates = []
    rf = eligible[eligible["model_key"].eq("random_forest")]
    if not rf.empty:
        horizon_candidates.append(rf.sort_values(["validation_1m_auc", "validation_1m_ks"], ascending=False).iloc[0])
    xgb = eligible[eligible["model_key"].eq("xgboost")]
    if not xgb.empty:
        horizon_candidates.extend(
            list(xgb.sort_values(["validation_1m_auc", "validation_1m_ks"], ascending=False).head(3).itertuples(index=False))
        )

    validation_12m = workflow_df[
        workflow_df[SPLIT_LABEL_COLUMN].eq("validation")
    ].copy()
    validation_raw = raw_loss_df[raw_loss_df[ID_COLUMN].isin(validation_12m[ID_COLUMN])].copy()
    actual = validation_12m.set_index(ID_COLUMN)["actual_default_12m"].astype(int)
    target_mean = float(actual.mean())

    best_key = None
    best_tuple = None
    for candidate in horizon_candidates:
        candidate_dict = candidate._asdict() if hasattr(candidate, "_asdict") else candidate.to_dict()
        key = (candidate_dict["model_key"], int(candidate_dict["candidate_id"]))
        bundle = bundles[key]
        horizon_start = time.perf_counter()
        pred = _score_hazard_12m(bundle, validation_raw, snapshot_mode="origination", calibration_shift=0.0)
        y = pred[ID_COLUMN].map(actual).fillna(0).astype(int)
        shift = _fit_intercept_shift(pred["predicted_pd_12m_raw"], target_mean)
        pred["predicted_pd_12m"] = _logit_shift(pred["predicted_pd_12m_raw"], shift)
        metrics = _safe_evaluate_model(y, pred["predicted_pd_12m"])
        horizon_seconds = time.perf_counter() - horizon_start
        mask = (
            search["model_key"].eq(candidate_dict["model_key"])
            & search["candidate_id"].eq(int(candidate_dict["candidate_id"]))
        )
        search.loc[mask, "validation_12m_auc"] = metrics["AUC"]
        search.loc[mask, "validation_12m_ks"] = metrics["KS"]
        search.loc[mask, "validation_12m_brier"] = metrics["Brier"]
        search.loc[mask, "horizon_score_seconds"] = horizon_seconds
        search.loc[mask, "pd_12m_calibration_shift"] = shift
        bundle["pd_12m_calibration"] = {
            "method": "post_aggregation_logit_intercept",
            "intercept_shift": shift,
            "target": "validation_12m_observable_default_rate",
            "validation_default_rate": target_mean,
            "validation_raw_mean": float(pred["predicted_pd_12m_raw"].mean()),
            "validation_calibrated_mean": float(pred["predicted_pd_12m"].mean()),
        }
        key_tuple = (float(metrics["AUC"]), float(metrics["KS"]), -float(metrics["Brier"]))
        if best_tuple is None or key_tuple > best_tuple:
            best_tuple = key_tuple
            best_key = key
        print(
            f"12M horizon {key[0]} #{key[1]} auc={metrics['AUC']:.4f} "
            f"ks={metrics['KS']:.4f} brier={metrics['Brier']:.4f} seconds={horizon_seconds:.1f}",
            flush=True,
        )

    if best_key is None:
        best_row = eligible.sort_values(["validation_1m_auc", "validation_1m_ks"], ascending=False).iloc[0]
        best_key = (best_row["model_key"], int(best_row["candidate_id"]))
        bundles[best_key]["pd_12m_calibration"] = {
            "method": "none",
            "intercept_shift": 0.0,
            "target": "not_fit_runtime_guard",
        }

    best_params = copy.deepcopy(
        search[
            search["model_key"].eq(best_key[0]) & search["candidate_id"].eq(best_key[1])
        ].iloc[0]["params_payload"]
    )
    train_validation = pd.concat([prepared["train"]["df"], prepared["validation"]["df"]], ignore_index=True)
    profile = _hazard_feature_profile_for_model(best_key[0], "calendar_time")
    final_payload = _build_calendar_hazard_panel_payload(
        train_validation,
        feature_profile=profile,
        meta_columns=prepared["meta_columns"],
        model_name=best_key[0],
    )
    final_bundle = _fit_hazard_model_from_payload(
        best_key[0],
        best_params,
        final_payload,
        prepared["meta_columns"],
        feature_mode="calendar_time",
    )
    final_bundle.update(
        {
            "modeling_type": PD_MODELING_TYPE_CALENDAR_TIME_HAZARD,
            "hazard_impute_values": prepared["hazard_impute_values"],
            "data_cutoff": prepared["data_cutoff"],
            "prepared_data": prepared,
            "stage2_scoring_chunk_rows": FUTURE_SCORE_CHUNK_ROWS,
            "pd_12m_calibration": copy.deepcopy(bundles[best_key].get("pd_12m_calibration", {})),
        }
    )
    final_validation_pred = _score_hazard_12m(
        final_bundle,
        validation_raw,
        snapshot_mode="origination",
        calibration_shift=0.0,
    )
    final_y = final_validation_pred[ID_COLUMN].map(actual).fillna(0).astype(int)
    final_shift = _fit_intercept_shift(final_validation_pred["predicted_pd_12m_raw"], target_mean)
    final_validation_pred["predicted_pd_12m"] = _logit_shift(
        final_validation_pred["predicted_pd_12m_raw"],
        final_shift,
    )
    final_metrics = _safe_evaluate_model(final_y, final_validation_pred["predicted_pd_12m"])
    final_bundle["pd_12m_calibration"] = {
        "method": "post_aggregation_logit_intercept",
        "intercept_shift": final_shift,
        "target": "final_refit_validation_12m_observable_default_rate",
        "validation_default_rate": target_mean,
        "validation_raw_mean": float(final_validation_pred["predicted_pd_12m_raw"].mean()),
        "validation_calibrated_mean": float(final_validation_pred["predicted_pd_12m"].mean()),
        "validation_auc": float(final_metrics["AUC"]),
        "validation_ks": float(final_metrics["KS"]),
        "validation_brier": float(final_metrics["Brier"]),
    }
    selected = search[
        search["model_key"].eq(best_key[0]) & search["candidate_id"].eq(best_key[1])
    ].iloc[0].to_dict()
    runtime = {
        "prepared_split_summary": prepared["split_summary"].to_dict(orient="records"),
        "skipped_candidates": [
            {
                "model_key": item["model_key"],
                "candidate_id": item["candidate_id"],
                "reason": item["reason"],
            }
            for item in skipped
        ],
        "selected_12m_model_key": best_key[0],
        "selected_12m_candidate_id": best_key[1],
        "selected_12m_params": best_params,
        "selected_12m_validation_auc": float(selected.get("validation_12m_auc", np.nan)),
        "selected_12m_validation_ks": float(selected.get("validation_12m_ks", np.nan)),
        "selected_12m_validation_brier": float(selected.get("validation_12m_brier", np.nan)),
        "final_refit_12m_validation_auc": float(final_metrics["AUC"]),
        "final_refit_12m_validation_ks": float(final_metrics["KS"]),
        "final_refit_12m_validation_brier": float(final_metrics["Brier"]),
    }
    return final_bundle, search, runtime


def _scope_raw(raw_loss_df, workflow_subset):
    ids = workflow_subset[ID_COLUMN].dropna().astype(int).unique()
    return raw_loss_df[raw_loss_df[ID_COLUMN].isin(ids)].copy()


def _score_dual_scope(
    static_bundle: dict,
    hazard_bundle: dict,
    raw_scope_df: pd.DataFrame,
    scope_name: str,
    snapshot_mode: str,
) -> pd.DataFrame:
    static_pred = _score_static_lifetime(static_bundle, raw_scope_df)
    hazard_pred = _score_hazard_12m(hazard_bundle, raw_scope_df, snapshot_mode=snapshot_mode)
    pred = static_pred.merge(hazard_pred, on=ID_COLUMN, how="outer")
    pred["prediction_scope"] = scope_name
    pred["predicted_pd"] = pred["predicted_pd_lifetime_static"].clip(0, 1)
    pred["predicted_pd_lifetime"] = pred["predicted_pd_lifetime_static"].clip(0, 1)
    pred["predicted_pd_12m"] = pred["predicted_pd_12m"].clip(0, 1)
    pred["model_name"] = "Dual PD: Static HGB lifetime + calendar hazard 12M"
    pred["hazard_12m_model_name"] = hazard_bundle["model_name"]
    return pred


def _build_stage2_predictions(static_bundle, hazard_bundle, raw_loss_df, workflow_df) -> pd.DataFrame:
    resolved = workflow_df[
        workflow_df[SPLIT_LABEL_COLUMN].eq("test") & workflow_df[STATUS_COLUMN].isin(RESOLVED_LOAN_STATUSES)
    ].copy()
    active = workflow_df[workflow_df[STATUS_COLUMN].isin(ACTIVE_LOAN_STATUSES)].copy()
    observable_12m = workflow_df[
        workflow_df[SPLIT_LABEL_COLUMN].eq("test")
        & (pd.to_datetime(workflow_df[ISSUE_DATE_COLUMN]) < pd.Timestamp("2018-01-01"))
    ].copy()
    scopes = [
        ("resolved_test", resolved, "origination"),
        ("active_snapshot", active, "active_cutoff"),
        ("observable_12m_test", observable_12m, "origination"),
    ]
    frames = []
    for scope_name, scope_df, mode in scopes:
        print(f"Scoring dual PD scope={scope_name} rows={len(scope_df):,}", flush=True)
        frames.append(
            _score_dual_scope(
                static_bundle,
                hazard_bundle,
                _scope_raw(raw_loss_df, scope_df),
                scope_name,
                mode,
            )
        )
    return pd.concat(frames, ignore_index=True)


def _write_stage2_config(static_summary, hazard_bundle, hazard_runtime, view_summary, runtime_seconds):
    config = {
        "modeling_type": "dual_pd",
        "training_mode": "dual_pd_30min_static_lifetime_hazard_12m",
        "data_cutoff": DATA_CUTOFF.strftime("%Y-%m-%d"),
        "split_date_column": {
            "lifetime": "issue_date",
            "12m": "snapshot_month",
        },
        "processed_data_path": str(DATA_DIR / "raw" / "accepted_2007_to_2018Q4.csv"),
        "precomputed_dual_pd_prediction_path": str(STAGE2_PREDICTION_PATH),
        "observable_12m_loss_metrics_path": str(OBSERVABLE_12M_METRICS_PATH),
        "selected_model_key": "dual_pd_static_hgb_hazard_12m",
        "selected_lifetime_model_key": "hist_gradient_boosting",
        "selected_12m_model_key": hazard_runtime["selected_12m_model_key"],
        "model_name": "Dual PD: Static HGB lifetime + calendar hazard 12M",
        "lifetime_model": static_summary,
        "hazard_12m_model": {
            "model_key": hazard_runtime["selected_12m_model_key"],
            "model_name": hazard_bundle["model_name"],
            "params": hazard_runtime["selected_12m_params"],
            "feature_profile": hazard_bundle["feature_profile"],
            "feature_count": len(hazard_bundle["feature_columns"]),
            "pd_12m_calibration": hazard_bundle.get("pd_12m_calibration", {}),
        },
        "calendar_modeling_sample_path": str(VIEW_PATH),
        "calendar_modeling_sample_summary_path": str(VIEW_SUMMARY_PATH),
        "view_summary": view_summary,
        "candidate_count_planned": 8,
        "candidate_count_completed": int(1 + len(pd.read_csv(SEARCH_TABLE_PATH))) if SEARCH_TABLE_PATH.exists() else None,
        "training_runtime_seconds": runtime_seconds,
        "full_stage2_scoring": True,
        "stage2_scoring_parallel_jobs": 1,
        "max_stage2_scoring_rows_per_scope": None,
        "random_state": RANDOM_STATE,
    }
    _write_json(STAGE2_CONFIG_PATH, config)
    return config


def _write_test_pd_export(static_bundle, hazard_bundle, raw_loss_df, workflow_df):
    test_12m = workflow_df[
        workflow_df[SPLIT_LABEL_COLUMN].eq("test")
        & (pd.to_datetime(workflow_df[ISSUE_DATE_COLUMN]) < pd.Timestamp("2018-01-01"))
    ].copy()
    pred = _score_dual_scope(
        static_bundle,
        hazard_bundle,
        _scope_raw(raw_loss_df, test_12m),
        "observable_12m_test",
        "origination",
    )
    export = test_12m.merge(pred, on=ID_COLUMN, how="left")
    keep = [
        ID_COLUMN,
        ISSUE_DATE_COLUMN,
        "loan_amnt",
        "annual_inc",
        "fico_range_low",
        "term_months",
        "int_rate",
        "revol_util",
        "actual_default_12m",
        "charged_off_flag",
        "predicted_hazard_1m",
        "predicted_pd_12m_raw",
        "predicted_pd_12m",
        "predicted_pd_lifetime_static_raw",
        "predicted_pd_lifetime_static",
        "predicted_pd",
        "model_name",
        "lifetime_model_name",
        "hazard_12m_model_name",
        "lifetime_model_source",
    ]
    export[[c for c in keep if c in export.columns]].to_csv(TEST_PD_PATH, index=False)
    export.to_csv(TEST_12M_BACKTEST_PATH, index=False)
    return export


def main():
    start_total = time.perf_counter()
    raw_path = DATA_DIR / "raw" / "accepted_2007_to_2018Q4.csv"
    print(f"Loading raw loss data from {raw_path}", flush=True)
    raw_loss_df = ensure_sample_id(load_loss_raw_dataset(raw_path))
    workflow_df = prepare_loss_workflow_dataset(raw_loss_df)

    print("Fitting GitHub original static lifetime HGB...", flush=True)
    static_bundle, static_summary = _fit_static_lifetime_model(raw_loss_df)
    print(json.dumps(static_summary, indent=2, default=_json_default), flush=True)

    print("Reading existing calendar modeling sample...", flush=True)
    modeling_sample = read_modeling_sample(PROCESSED_DIR / "calendar_survival_modeling_sample.parquet")
    view, view_summary = _build_training_view(modeling_sample)
    VIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    view.to_parquet(VIEW_PATH, index=False)
    _write_json(VIEW_SUMMARY_PATH, view_summary)
    print(json.dumps(view_summary, indent=2, default=_json_default), flush=True)

    print("Fitting 12M calendar-time hazard search...", flush=True)
    hazard_bundle, search_table, hazard_runtime = _fit_12m_hazard_model(
        view,
        workflow_df,
        raw_loss_df,
        start_total,
    )
    search_table.drop(columns=["params_payload"], errors="ignore").to_csv(SEARCH_TABLE_PATH, index=False)

    print("Scoring validation/test 12M observable backtests...", flush=True)
    validation_12m = workflow_df[workflow_df[SPLIT_LABEL_COLUMN].eq("validation")].copy()
    validation_pred = _score_dual_scope(
        static_bundle,
        hazard_bundle,
        _scope_raw(raw_loss_df, validation_12m),
        "validation_12m_observable",
        "origination",
    )
    validation_12m.merge(validation_pred, on=ID_COLUMN, how="left").to_csv(
        VALIDATION_12M_BACKTEST_PATH,
        index=False,
    )
    test_export = _write_test_pd_export(static_bundle, hazard_bundle, raw_loss_df, workflow_df)
    resolved_lifetime = workflow_df[
        workflow_df[SPLIT_LABEL_COLUMN].eq("test") & workflow_df[STATUS_COLUMN].isin(RESOLVED_LOAN_STATUSES)
    ].copy()
    resolved_lifetime.merge(
        _score_static_lifetime(static_bundle, _scope_raw(raw_loss_df, resolved_lifetime)),
        on=ID_COLUMN,
        how="left",
    ).to_csv(LIFETIME_DIAGNOSTIC_PATH, index=False)

    print("Scoring full stage2 dual PD scopes...", flush=True)
    stage2_predictions = _build_stage2_predictions(
        static_bundle,
        hazard_bundle,
        raw_loss_df,
        workflow_df,
    )
    stage2_predictions.to_csv(STAGE2_PREDICTION_PATH, index=False)

    runtime_seconds = float(time.perf_counter() - start_total)
    stage2_config = _write_stage2_config(
        static_summary,
        hazard_bundle,
        hazard_runtime,
        view_summary,
        runtime_seconds,
    )
    runtime_payload = {
        "training_mode": "dual_pd_30min_static_lifetime_hazard_12m",
        "runtime_seconds_including_full_stage2_prediction_export": runtime_seconds,
        "training_budget_seconds": TRAINING_BUDGET_SECONDS,
        "static_lifetime": static_summary,
        "hazard_12m": hazard_runtime,
        "search_table_path": str(SEARCH_TABLE_PATH),
        "stage2_prediction_path": str(STAGE2_PREDICTION_PATH),
        "stage2_config_path": str(STAGE2_CONFIG_PATH),
        "test_pd_path": str(TEST_PD_PATH),
    }
    _write_json(RUNTIME_PATH, runtime_payload)

    print("Running loss reserve workflow from precomputed dual PD predictions...", flush=True)
    loss_config = build_loss_config(stage2_config)
    loss_config["observable_12m_loss_metrics_path"] = str(OBSERVABLE_12M_METRICS_PATH)
    loss_result = run_loss_reserve_workflow(loss_config)
    export_loss_reserve_visuals(
        loss_result,
        output_dir=REPORT_DIR / "figures" / "loss_reserve",
        show=False,
    )
    print(
        json.dumps(
            {
                "static_lifetime_test_auc": static_summary["test_auc_calibrated"],
                "selected_12m_model": hazard_runtime["selected_12m_model_key"],
                "selected_12m_validation_auc": hazard_runtime["selected_12m_validation_auc"],
                "test_12m_rows": int(len(test_export)),
                "runtime_seconds": runtime_seconds,
                "portfolio_summary_path": str(PROCESSED_DIR / "portfolio_expected_loss_summary.csv"),
            },
            indent=2,
            default=_json_default,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
