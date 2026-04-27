from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from calendar_bigdata import read_modeling_sample
from experiment import (
    FULL_CALENDAR_HAZARD_PROFILE,
    ID_COLUMN,
    MODEL_LABELS,
    PD_MODELING_TYPE_CALENDAR_TIME_HAZARD,
    SNAPSHOT_MONTH_COLUMN,
    TARGET_1M_COLUMN,
    _add_calendar_time_features,
    _build_calendar_hazard_panel_payload,
    _build_hazard_design_matrix,
    _build_importance_tables,
    _fit_hazard_model_from_payload,
    _hazard_feature_profile_for_model,
    _metric_comparison_key,
    _safe_evaluate_model,
    prepare_experiment_data,
)
from loss_workflow import run_loss_reserve_workflow
from reporting import write_temporal_report
from run_hazard_pd_pipeline import PROCESSED_DIR, REPORT_DIR, build_loss_config, build_pd_config
from visualization import export_loss_reserve_visuals, export_temporal_visuals


RANDOM_STATE = 42
TRAIN_CAP = 250_000
TRAIN_NEGATIVE_TO_POSITIVE_RATIO = 3
VALIDATION_CAP = 25_000
TEST_CAP = 100_000
SCORING_CAP = 25_000
MLP_SKIP_AFTER_SECONDS = 25 * 60
FUTURE_SCORE_CHUNK_ROWS = 4_000

VIEW_PATH = PROCESSED_DIR / "calendar_survival_30min_training_view.parquet"
VIEW_SUMMARY_PATH = PROCESSED_DIR / "calendar_survival_30min_training_view_summary.json"
RUNTIME_PATH = PROCESSED_DIR / "calendar_30min_training_runtime.json"
SEARCH_TABLE_PATH = PROCESSED_DIR / "calendar_30min_grid_search_results.csv"
VALIDATION_BEST_PATH = PROCESSED_DIR / "calendar_30min_model_best_validation.csv"


RF_CANDIDATES = [
    {"n_estimators": 60, "max_depth": 8, "min_samples_leaf": 75, "max_samples": 0.6},
    {"n_estimators": 80, "max_depth": 10, "min_samples_leaf": 50, "max_samples": 0.7},
    {"n_estimators": 90, "max_depth": 12, "min_samples_leaf": 50, "max_samples": 0.7},
]
XGB_CANDIDATES = [
    {"n_estimators": 80, "max_depth": 4, "learning_rate": 0.08},
    {"n_estimators": 80, "max_depth": 6, "learning_rate": 0.05},
    {"n_estimators": 120, "max_depth": 4, "learning_rate": 0.05},
    {"n_estimators": 120, "max_depth": 6, "learning_rate": 0.08},
]
MLP_CANDIDATES = [
    {
        "hidden_layer_sizes": (64, 32, 16),
        "max_iter": 30,
        "batch_size": 4096,
        "early_stopping": True,
        "validation_fraction": 0.05,
        "n_iter_no_change": 3,
        "alpha": 0.0001,
        "learning_rate_init": 0.001,
    }
]


def _json_default(value):
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def _params_json(params: dict) -> str:
    return json.dumps(params, sort_keys=True, default=_json_default)


def _sample_partition(
    partition: pd.DataFrame,
    *,
    max_rows: int,
    random_state: int,
    preserve_positive: bool = False,
    negative_to_positive_ratio: int | None = None,
) -> pd.DataFrame:
    if partition.empty or len(partition) <= max_rows:
        return partition.copy()
    positive = partition[partition[TARGET_1M_COLUMN].fillna(0).astype(int).eq(1)]
    negative = partition[partition[TARGET_1M_COLUMN].fillna(0).astype(int).eq(0)]
    if preserve_positive:
        positive_sample = positive.copy()
        negative_cap = max(0, max_rows - len(positive_sample))
        if negative_to_positive_ratio is not None and not positive_sample.empty:
            negative_cap = min(negative_cap, len(positive_sample) * negative_to_positive_ratio)
        negative_sample = negative.sample(
            n=min(len(negative), negative_cap),
            random_state=random_state,
        )
    else:
        positive_n = int(round(max_rows * len(positive) / max(len(partition), 1)))
        if not positive.empty:
            positive_n = max(1, positive_n)
        positive_n = min(len(positive), positive_n)
        negative_n = min(len(negative), max_rows - positive_n)
        positive_sample = positive.sample(n=positive_n, random_state=random_state)
        negative_sample = negative.sample(n=negative_n, random_state=random_state + 1)
    return pd.concat([positive_sample, negative_sample], ignore_index=True)


def _build_30min_view(modeling_sample: pd.DataFrame) -> pd.DataFrame:
    partitions = []
    for split_name, cap in [
        ("train", TRAIN_CAP),
        ("validation", VALIDATION_CAP),
        ("test", TEST_CAP),
    ]:
        partition = modeling_sample[modeling_sample["split_label"] == split_name].copy()
        sampled = _sample_partition(
            partition,
            max_rows=cap,
            random_state=RANDOM_STATE + len(partitions) * 17,
            preserve_positive=(split_name == "train"),
            negative_to_positive_ratio=TRAIN_NEGATIVE_TO_POSITIVE_RATIO
            if split_name == "train"
            else None,
        )
        partitions.append(sampled)

    scoring = modeling_sample[modeling_sample["split_label"] == "scoring"].copy()
    if len(scoring) > SCORING_CAP:
        scoring = scoring.sample(n=SCORING_CAP, random_state=RANDOM_STATE + 99)
    partitions.append(scoring)

    view = pd.concat(partitions, ignore_index=True)
    sort_columns = [column for column in ["split_label", SNAPSHOT_MONTH_COLUMN, ID_COLUMN] if column in view.columns]
    view = view.sort_values(sort_columns).reset_index(drop=True)
    return view


def _write_view_summary(view: pd.DataFrame, source_path: Path) -> dict:
    summary = {
        "source_modeling_sample_path": str(source_path),
        "view_path": str(VIEW_PATH),
        "rows": int(len(view)),
        "split_counts": {
            str(key): int(value) for key, value in view["split_label"].value_counts().items()
        },
        "event_counts": {
            str(key): int(value)
            for key, value in view.groupby("split_label")[TARGET_1M_COLUMN].sum(min_count=1).fillna(0).items()
        },
        "caps": {
            "train": TRAIN_CAP,
            "train_negative_to_positive_ratio": TRAIN_NEGATIVE_TO_POSITIVE_RATIO,
            "validation": VALIDATION_CAP,
            "test": TEST_CAP,
            "scoring": SCORING_CAP,
        },
        "sample_id_snapshot_unique": bool(
            not view.duplicated([ID_COLUMN, SNAPSHOT_MONTH_COLUMN]).any()
        ),
        "random_state": RANDOM_STATE,
    }
    VIEW_SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return summary


def _build_30min_config(view_path: Path) -> dict:
    config = build_pd_config()
    config.update(
        {
            "calendar_panel_is_prebuilt": True,
            "calendar_panel_path": None,
            "calendar_panel_counts_path": None,
            "calendar_modeling_sample_path": str(view_path),
            "calendar_modeling_sample_summary_path": str(VIEW_SUMMARY_PATH),
            "enabled_models": [
                "logistic_regression",
                "random_forest",
                "xgboost",
                "mlp_neural_network",
            ],
            "grid_search_models": ["random_forest", "xgboost"],
            "use_grid_search": True,
            "thirty_minute_training": True,
            "champion_eligible_models": ["random_forest", "xgboost"],
            "output_path": str(PROCESSED_DIR / "test_with_pd_best_model.csv"),
            "stage2_champion_config_path": str(PROCESSED_DIR / "stage2_champion_config.json"),
            "report_output_path": str(REPORT_DIR / "temporal_model_report.md"),
        }
    )
    config["model_params"]["logistic_regression"].update({"max_iter": 300, "C": 1.0})
    config["model_params"]["random_forest"].update(
        {
            "n_estimators": 80,
            "max_depth": 10,
            "min_samples_split": 200,
            "min_samples_leaf": 50,
            "max_features": "sqrt",
            "bootstrap": True,
            "max_samples": 0.7,
            "n_jobs": -1,
            "random_state": RANDOM_STATE,
        }
    )
    config["model_params"]["xgboost"].update(
        {
            "n_estimators": 120,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "reg_lambda": 1.0,
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        }
    )
    config["model_params"]["mlp_neural_network"].update(MLP_CANDIDATES[0])
    config["mlp_sampling"].update(
        {
            "negative_to_positive_ratio": TRAIN_NEGATIVE_TO_POSITIVE_RATIO,
            "max_train_rows": TRAIN_CAP,
            "random_state": RANDOM_STATE,
        }
    )
    config["param_grid"] = {
        "logistic_regression": {},
        "random_forest": {},
        "xgboost": {},
        "mlp_neural_network": {},
    }
    return config


def _candidate_plan(config: dict) -> list[dict]:
    return [
        {
            "candidate_id": 1,
            "model_key": "logistic_regression",
            "params": copy.deepcopy(config["model_params"]["logistic_regression"]),
            "champion_eligible": False,
        },
        *[
            {
                "candidate_id": index + 1,
                "model_key": "random_forest",
                "params": {
                    **copy.deepcopy(config["model_params"]["random_forest"]),
                    **override,
                },
                "champion_eligible": True,
            }
            for index, override in enumerate(RF_CANDIDATES)
        ],
        *[
            {
                "candidate_id": index + 1,
                "model_key": "xgboost",
                "params": {
                    **copy.deepcopy(config["model_params"]["xgboost"]),
                    **override,
                },
                "champion_eligible": True,
            }
            for index, override in enumerate(XGB_CANDIDATES)
        ],
        *[
            {
                "candidate_id": index + 1,
                "model_key": "mlp_neural_network",
                "params": {
                    **copy.deepcopy(config["model_params"]["mlp_neural_network"]),
                    **override,
                },
                "champion_eligible": False,
                "optional_smoke": True,
            }
            for index, override in enumerate(MLP_CANDIDATES)
        ],
    ]


def _fast_score_model(model_bundle: dict, scoring_df: pd.DataFrame) -> pd.DataFrame:
    design = _build_hazard_design_matrix(
        scoring_df,
        feature_profile=model_bundle["feature_profile"],
        design_columns=model_bundle["feature_columns"],
    )
    predicted_hazard = model_bundle["model"].predict_proba(design)[:, 1].clip(0, 1)
    meta_columns = [
        column for column in model_bundle.get("meta_columns", []) if column in scoring_df.columns
    ]
    prediction = scoring_df[meta_columns].copy() if meta_columns else pd.DataFrame(index=scoring_df.index)
    prediction["actual_default"] = scoring_df[TARGET_1M_COLUMN].fillna(0).astype(int).to_numpy()
    prediction["actual_default_12m"] = (
        scoring_df.get("target_12m", scoring_df[TARGET_1M_COLUMN]).fillna(0).astype(int).to_numpy()
    )
    prediction["predicted_hazard_1m"] = predicted_hazard
    prediction["predicted_pd"] = predicted_hazard
    prediction["predicted_pd_12m"] = predicted_hazard
    prediction["hazard_model_key"] = model_bundle["model_key"]
    prediction["hazard_model_name"] = model_bundle["model_name"]
    prediction["model_name"] = model_bundle["model_name"]
    return prediction.reset_index(drop=True)


def _build_future_rows_vectorized(scoring_chunk: pd.DataFrame, row_ids: np.ndarray) -> pd.DataFrame:
    remaining = (
        pd.to_numeric(scoring_chunk["remaining_term"], errors="coerce")
        .fillna(0)
        .round()
        .clip(lower=0)
        .astype(int)
        .to_numpy()
    )
    valid_mask = remaining > 0
    if not valid_mask.any():
        return pd.DataFrame()

    base = scoring_chunk.loc[valid_mask].reset_index(drop=True)
    base_row_ids = row_ids[valid_mask]
    counts = remaining[valid_mask]
    repeated_positions = np.repeat(np.arange(len(base)), counts)
    horizons = np.concatenate(
        [np.arange(1, count + 1, dtype=np.int16) for count in counts]
    )
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
        pd.to_numeric(base["month_on_book"], errors="coerce")
        .fillna(0)
        .to_numpy()
    )
    term_months = pd.to_numeric(base["term_months"], errors="coerce").fillna(0).to_numpy()
    future["month_on_book"] = np.repeat(base_month_on_book, counts) + horizons - 1
    future["remaining_term"] = np.maximum(
        np.repeat(term_months, counts) - future["month_on_book"],
        0,
    )
    future[TARGET_1M_COLUMN] = 0
    return _add_calendar_time_features(future)


def _score_calendar_time_hazard_model_bundle_chunked(
    model_bundle: dict,
    panel_df: pd.DataFrame,
    output_column: str = "predicted_pd",
) -> pd.DataFrame:
    scoring = panel_df.copy().reset_index(drop=True)
    if scoring.empty:
        return scoring.assign(
            predicted_hazard_1m=pd.Series(dtype=float),
            predicted_pd_12m=pd.Series(dtype=float),
            **{output_column: pd.Series(dtype=float)},
        )

    current_prediction = _fast_score_model(model_bundle, scoring)
    current_prediction["_score_row_id"] = np.arange(len(scoring))
    score_frames = []
    row_ids = np.arange(len(scoring))
    for start in range(0, len(scoring), FUTURE_SCORE_CHUNK_ROWS):
        end = min(start + FUTURE_SCORE_CHUNK_ROWS, len(scoring))
        future_rows = _build_future_rows_vectorized(scoring.iloc[start:end], row_ids[start:end])
        if future_rows.empty:
            continue
        future_design = _build_hazard_design_matrix(
            future_rows,
            feature_profile=model_bundle["feature_profile"],
            design_columns=model_bundle["feature_columns"],
        )
        future_rows["hazard_prob"] = (
            model_bundle["model"].predict_proba(future_design)[:, 1].clip(0, 1)
        )
        future_rows["survival_factor"] = 1.0 - future_rows["hazard_prob"].clip(0, 1)
        lifetime = (
            future_rows.groupby("_score_row_id", as_index=False)["survival_factor"]
            .prod()
            .rename(columns={"survival_factor": "survival_lifetime"})
        )
        within_12 = future_rows[future_rows["horizon_month"] <= 12]
        pd_12m = (
            within_12.groupby("_score_row_id", as_index=False)["survival_factor"]
            .prod()
            .rename(columns={"survival_factor": "survival_12m"})
        )
        scores = lifetime.merge(pd_12m, on="_score_row_id", how="left")
        scores["survival_12m"] = scores["survival_12m"].fillna(1.0)
        scores[output_column] = (1.0 - scores["survival_lifetime"]).clip(0, 1)
        scores["predicted_pd_12m"] = (1.0 - scores["survival_12m"]).clip(0, 1)
        score_frames.append(scores[["_score_row_id", output_column, "predicted_pd_12m"]])

    if score_frames:
        pd_scores = pd.concat(score_frames, ignore_index=True)
    else:
        pd_scores = pd.DataFrame(columns=["_score_row_id", output_column, "predicted_pd_12m"])
    prediction = current_prediction.drop(columns=[output_column, "predicted_pd_12m"], errors="ignore")
    prediction = prediction.merge(pd_scores, on="_score_row_id", how="left")
    prediction[output_column] = prediction[output_column].fillna(0.0).clip(0, 1)
    prediction["predicted_pd_12m"] = prediction["predicted_pd_12m"].fillna(0.0).clip(0, 1)
    prediction = prediction.drop(columns=["_score_row_id"], errors="ignore")
    return prediction


def _fit_candidate(model_key: str, params: dict, prepared: dict, payload_cache: dict) -> dict:
    if model_key not in payload_cache:
        feature_profile = _hazard_feature_profile_for_model(model_key, "calendar_time")
        payload_cache[model_key] = _build_calendar_hazard_panel_payload(
            prepared["train"]["df"],
            feature_profile=feature_profile,
            meta_columns=prepared["meta_columns"],
            mlp_sampling_config=prepared.get("mlp_sampling", {}),
            model_name=model_key,
        )
    payload = payload_cache[model_key]
    model_bundle = _fit_hazard_model_from_payload(
        model_key,
        params,
        payload,
        prepared["meta_columns"],
        feature_mode="calendar_time",
    )
    model_bundle.update(
        {
            "modeling_type": PD_MODELING_TYPE_CALENDAR_TIME_HAZARD,
            "hazard_impute_values": prepared["hazard_impute_values"],
            "data_cutoff": prepared["data_cutoff"],
            "prepared_data": prepared,
        }
    )
    return model_bundle


def _best_by_model(search_rows: list[dict]) -> dict[str, dict]:
    best = {}
    for row in search_rows:
        key = row["model_key"]
        if key not in best or _metric_comparison_key(row["validation_metrics"]) > _metric_comparison_key(
            best[key]["validation_metrics"]
        ):
            best[key] = row
    return best


def _prediction_export_columns(prediction: pd.DataFrame) -> list[str]:
    return [
        column
        for column in [
            "sample_id",
            "snapshot_month",
            "issue_date",
            "loan_amnt",
            "annual_inc",
            "fico_range_low",
            "term_months",
            "month_on_book",
            "remaining_term",
            "int_rate",
            "revol_util",
            "target_1m",
            "actual_default",
            "actual_default_12m",
            "predicted_hazard_1m",
            "predicted_pd_12m",
            "predicted_pd",
            "hazard_model_key",
            "hazard_model_name",
            "model_name",
        ]
        if column in prediction.columns
    ]


def _period_row(period: str, fitted_on: str, df: pd.DataFrame, prediction: pd.DataFrame) -> dict:
    metrics = _safe_evaluate_model(
        prediction["actual_default"],
        prediction["predicted_hazard_1m"],
    )
    return {
        "period": period,
        "fitted_on": fitted_on,
        "rows": int(len(df)),
        "default_rate": float(df[TARGET_1M_COLUMN].mean()),
        "min_snapshot_month": df[SNAPSHOT_MONTH_COLUMN].min(),
        "max_snapshot_month": df[SNAPSHOT_MONTH_COLUMN].max(),
        "auc": metrics["AUC"],
        "ks": metrics["KS"],
        "brier": metrics["Brier"],
    }


def _write_stage2_config(config: dict, result: dict, view_summary: dict, runtime: dict) -> dict:
    selected_model = result["selected_model_key"]
    final_model = result["final_models"][selected_model]
    stage2_config = {
        "modeling_type": "calendar_time_hazard",
        "training_mode": "30min_reuse_sample_xgb_rf_focus",
        "data_cutoff": config["data_cutoff"],
        "split_date_column": "snapshot_month",
        "calendar_panel_is_prebuilt": True,
        "calendar_modeling_sample_path": str(VIEW_PATH),
        "calendar_modeling_sample_summary_path": str(VIEW_SUMMARY_PATH),
        "processed_data_path": config["processed_data_path"],
        "random_state": RANDOM_STATE,
        "sample_frac": 1.0,
        "temporal_split": copy.deepcopy(config["temporal_split"]),
        "enabled_models": copy.deepcopy(config["enabled_models"]),
        "selected_model_key": selected_model,
        "meta_columns": copy.deepcopy(config["meta_columns"]),
        "model_params": {
            selected_model: copy.deepcopy(result["best_params_by_model"][selected_model])
        },
        "use_grid_search": False,
        "grid_search_models": [],
        "param_grid": {},
        "feature_profile": final_model.get("feature_profile"),
        "feature_count": len(final_model.get("feature_columns", [])),
        "pd_output": "predicted_pd_lifetime_calendar_time_hazard",
        "pd_12m_output": "predicted_pd_12m",
        "hazard_feature_mode": "calendar_time",
        "hazard_impute_values": result["prepared_data"]["hazard_impute_values"],
        "mlp_sampling": copy.deepcopy(config["mlp_sampling"]),
        "champion_eligible_models": ["random_forest", "xgboost"],
        "candidate_count": int(runtime["candidate_count_ran"]),
        "candidate_count_planned": int(runtime["candidate_count_planned"]),
        "validation_cap": VALIDATION_CAP,
        "train_cap": TRAIN_CAP,
        "test_cap": TEST_CAP,
        "scoring_cap": SCORING_CAP,
        "view_summary": view_summary,
        "training_runtime_seconds": runtime["total_seconds"],
        "full_stage2_scoring": True,
        "stage2_scoring_chunk_rows": 4000,
        "stage2_scoring_parallel_jobs": 2,
        "stage2_time_estimate_path": str(PROCESSED_DIR / "stage2_full_scoring_time_estimate.json"),
        "max_stage2_scoring_rows_per_scope": None,
    }
    Path(config["stage2_champion_config_path"]).write_text(
        json.dumps(stage2_config, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return stage2_config


def _append_30min_report_note(report_path: Path, result: dict, runtime: dict) -> None:
    lines = [
        "",
        "## 30-Minute Training Run Notes",
        "",
        "- This run reuses the existing calendar-time modeling sample and does not rebuild the raw Dask sample.",
        f"- Train/validation/test caps: `{TRAIN_CAP:,}` / `{VALIDATION_CAP:,}` / `{TEST_CAP:,}`.",
        "- Grid search uses fast 1-month hazard scoring; lifetime and 12-month PD aggregation is limited to RF/XGB model-best candidates and the final champion.",
        f"- Planned candidates: `{runtime['candidate_count_planned']}`; ran candidates: `{runtime['candidate_count_ran']}`.",
        f"- Total training/report runtime before downstream loss workflow: `{runtime['total_seconds']:.1f}` seconds.",
        f"- Selected champion: `{result['selected_model_key']}`.",
    ]
    with report_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> None:
    start_total = time.perf_counter()
    source_path = PROCESSED_DIR / "calendar_survival_modeling_sample.parquet"
    print(f"Reading existing modeling sample: {source_path}", flush=True)
    modeling_sample = read_modeling_sample(source_path)

    print("Building deterministic 30-minute training view...", flush=True)
    view = _build_30min_view(modeling_sample)
    VIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    view.to_parquet(VIEW_PATH, index=False)
    view_summary = _write_view_summary(view, source_path)
    print(json.dumps(view_summary, indent=2, default=_json_default), flush=True)

    config = _build_30min_config(VIEW_PATH)
    prepared = prepare_experiment_data(config, view)
    prepared["mlp_sampling"] = copy.deepcopy(config["mlp_sampling"])
    print(prepared["split_summary"].to_string(index=False), flush=True)

    candidates = _candidate_plan(config)
    payload_cache = {}
    search_rows = []
    model_bundles = {}
    validation_predictions = {}
    runtime_rows = []

    for candidate in candidates:
        if candidate.get("optional_smoke") and time.perf_counter() - start_total > MLP_SKIP_AFTER_SECONDS:
            print("Skipping optional MLP smoke candidate because runtime exceeded 25 minutes.", flush=True)
            continue
        model_key = candidate["model_key"]
        params = copy.deepcopy(candidate["params"])
        print(
            f"Candidate {model_key} #{candidate['candidate_id']} params={_params_json(params)}",
            flush=True,
        )
        fit_start = time.perf_counter()
        model_bundle = _fit_candidate(model_key, params, prepared, payload_cache)
        fit_seconds = time.perf_counter() - fit_start

        score_start = time.perf_counter()
        validation_prediction = _fast_score_model(model_bundle, prepared["validation"]["df"])
        score_seconds = time.perf_counter() - score_start
        metrics_1m = _safe_evaluate_model(
            validation_prediction["actual_default"],
            validation_prediction["predicted_hazard_1m"],
        )
        row = {
            "candidate_id": candidate["candidate_id"],
            "model_key": model_key,
            "model_name": MODEL_LABELS[model_key],
            "champion_eligible": bool(candidate.get("champion_eligible", False)),
            "feature_profile": model_bundle["feature_profile"],
            "feature_count": len(model_bundle["feature_columns"]),
            "search_mode": "30min_fast_1m_hazard_grid",
            "params": _params_json(params),
            "params_payload": params,
            "validation_auc": metrics_1m["AUC"],
            "validation_ks": metrics_1m["KS"],
            "validation_brier": metrics_1m["Brier"],
            "validation_12m_auc": np.nan,
            "validation_12m_brier": np.nan,
            "validation_lifetime_auc": np.nan,
            "validation_lifetime_brier": np.nan,
            "calibration_summary": metrics_1m["calibration_summary"],
            "validation_metrics": metrics_1m,
            "validation_metrics_12m": metrics_1m,
            "fit_seconds": fit_seconds,
            "score_seconds": score_seconds,
        }
        search_rows.append(row)
        runtime_rows.append(
            {
                "model_key": model_key,
                "candidate_id": candidate["candidate_id"],
                "fit_seconds": fit_seconds,
                "score_seconds": score_seconds,
                "validation_auc": metrics_1m["AUC"],
                "params": params,
            }
        )

        best_so_far = _best_by_model(search_rows)[model_key]
        if best_so_far is row:
            model_bundles[model_key] = model_bundle
            validation_predictions[model_key] = validation_prediction
        print(
            f"done fit={fit_seconds:.1f}s score={score_seconds:.1f}s "
            f"auc={metrics_1m['AUC']:.4f} ks={metrics_1m['KS']:.4f}",
            flush=True,
        )

    best_rows_by_model = _best_by_model(search_rows)
    for model_key in ["random_forest", "xgboost"]:
        print(f"Computing full validation PD for {model_key} best candidate...", flush=True)
        full_start = time.perf_counter()
        full_prediction = _score_calendar_time_hazard_model_bundle_chunked(
            model_bundles[model_key],
            prepared["validation"]["df"],
        )
        full_seconds = time.perf_counter() - full_start
        metrics_1m = _safe_evaluate_model(
            full_prediction["actual_default"],
            full_prediction["predicted_hazard_1m"],
        )
        metrics_12m = _safe_evaluate_model(
            full_prediction["actual_default_12m"],
            full_prediction["predicted_pd_12m"],
        )
        lifetime_metrics = _safe_evaluate_model(
            full_prediction["actual_default_12m"],
            full_prediction["predicted_pd"],
        )
        best_row = best_rows_by_model[model_key]
        best_row.update(
            {
                "validation_auc": metrics_1m["AUC"],
                "validation_ks": metrics_1m["KS"],
                "validation_brier": metrics_1m["Brier"],
                "validation_12m_auc": metrics_12m["AUC"],
                "validation_12m_brier": metrics_12m["Brier"],
                "validation_lifetime_auc": lifetime_metrics["AUC"],
                "validation_lifetime_brier": lifetime_metrics["Brier"],
                "validation_metrics": metrics_1m,
                "validation_metrics_12m": metrics_12m,
                "validation_metrics_lifetime": lifetime_metrics,
                "full_validation_score_seconds": full_seconds,
            }
        )
        validation_predictions[model_key] = full_prediction
        print(
            f"{model_key} full validation lifetime_auc={lifetime_metrics['AUC']:.4f} "
            f"12m_auc={metrics_12m['AUC']:.4f} seconds={full_seconds:.1f}",
            flush=True,
        )

    eligible_best = [best_rows_by_model[key] for key in ["random_forest", "xgboost"]]
    champion_row = max(
        eligible_best,
        key=lambda row: (
            float(row["validation_lifetime_auc"]),
            float(row["validation_ks"]),
            -float(row["validation_lifetime_brier"]),
        ),
    )
    champion_key = champion_row["model_key"]
    print(f"Selected champion: {champion_key}", flush=True)

    train_validation = pd.concat(
        [prepared["train"]["df"], prepared["validation"]["df"]],
        ignore_index=True,
    )
    feature_profile = _hazard_feature_profile_for_model(champion_key, "calendar_time")
    final_payload = _build_calendar_hazard_panel_payload(
        train_validation,
        feature_profile=feature_profile,
        meta_columns=prepared["meta_columns"],
        mlp_sampling_config=config.get("mlp_sampling", {}),
        model_name=champion_key,
    )
    final_fit_start = time.perf_counter()
    final_model = _fit_hazard_model_from_payload(
        champion_key,
        champion_row["params_payload"],
        final_payload,
        prepared["meta_columns"],
        feature_mode="calendar_time",
    )
    final_fit_seconds = time.perf_counter() - final_fit_start
    final_model.update(
        {
            "modeling_type": PD_MODELING_TYPE_CALENDAR_TIME_HAZARD,
            "hazard_impute_values": prepared["hazard_impute_values"],
            "data_cutoff": prepared["data_cutoff"],
            "prepared_data": prepared,
        }
    )

    print("Scoring champion on capped test sample with full PD aggregation...", flush=True)
    test_score_start = time.perf_counter()
    test_prediction = _score_calendar_time_hazard_model_bundle_chunked(
        final_model,
        prepared["test"]["df"],
    )
    test_score_seconds = time.perf_counter() - test_score_start
    test_metrics = _safe_evaluate_model(
        test_prediction["actual_default"],
        test_prediction["predicted_hazard_1m"],
    )
    test_metrics_12m = _safe_evaluate_model(
        test_prediction["actual_default_12m"],
        test_prediction["predicted_pd_12m"],
    )
    test_lifetime_metrics = _safe_evaluate_model(
        test_prediction["actual_default_12m"],
        test_prediction["predicted_pd"],
    )
    test_prediction[_prediction_export_columns(test_prediction)].to_csv(
        config["output_path"],
        index=False,
    )

    train_fast_prediction = _fast_score_model(model_bundles[champion_key], prepared["train"]["df"])
    validation_table_rows = []
    for model_key, row in best_rows_by_model.items():
        validation_table_rows.append(
            {
                "model_key": model_key,
                "model_name": MODEL_LABELS[model_key],
                "params": row["params"],
                "feature_profile": row["feature_profile"],
                "feature_count": row["feature_count"],
                "search_mode": row["search_mode"],
                "search_candidates": len(
                    [candidate_row for candidate_row in search_rows if candidate_row["model_key"] == model_key]
                ),
                "validation_auc": row["validation_auc"],
                "validation_ks": row["validation_ks"],
                "validation_brier": row["validation_brier"],
                "validation_12m_auc": row["validation_12m_auc"],
                "validation_12m_brier": row["validation_12m_brier"],
                "validation_lifetime_auc": row["validation_lifetime_auc"],
                "validation_lifetime_brier": row["validation_lifetime_brier"],
                "calibration_summary": row["calibration_summary"],
                "validation_metrics": row["validation_metrics"],
                "validation_metrics_12m": row["validation_metrics_12m"],
            }
        )
    validation_table = pd.DataFrame(validation_table_rows)
    validation_table["_rank_group"] = validation_table["model_key"].map(
        {"xgboost": 0, "random_forest": 0}
    ).fillna(1)
    validation_table = validation_table.sort_values(
        ["_rank_group", "validation_lifetime_auc", "validation_auc", "validation_ks"],
        ascending=[True, False, False, False],
    ).drop(columns=["_rank_group"]).reset_index(drop=True)
    validation_table["validation_rank"] = range(1, len(validation_table) + 1)

    final_test_table = pd.DataFrame(
        [
            {
                "model_key": champion_key,
                "model_name": MODEL_LABELS[champion_key],
                "validation_rank": 1,
                "params": _params_json(champion_row["params_payload"]),
                "feature_profile": final_model["feature_profile"],
                "feature_count": len(final_model["feature_columns"]),
                "search_mode": "refit_on_30min_train_plus_validation",
                "test_auc": test_metrics["AUC"],
                "test_ks": test_metrics["KS"],
                "test_brier": test_metrics["Brier"],
                "test_12m_auc": test_metrics_12m["AUC"],
                "test_12m_brier": test_metrics_12m["Brier"],
                "test_lifetime_auc": test_lifetime_metrics["AUC"],
                "test_lifetime_brier": test_lifetime_metrics["Brier"],
                "calibration_summary": test_metrics["calibration_summary"],
                "test_metrics": test_metrics,
                "test_metrics_12m": test_metrics_12m,
            }
        ]
    )
    champion_period_table = pd.DataFrame(
        [
            _period_row("train", "train only", prepared["train"]["df"], train_fast_prediction),
            _period_row(
                "validation",
                "train only",
                prepared["validation"]["df"],
                validation_predictions[champion_key],
            ),
            _period_row("test", "train + validation", prepared["test"]["df"], test_prediction),
        ]
    ).assign(model_name=MODEL_LABELS[champion_key])

    final_models = {champion_key: final_model}
    result = {
        "validation_table": validation_table,
        "final_test_table": final_test_table,
        "champion_validation": validation_table[validation_table["model_key"] == champion_key].iloc[0].to_dict(),
        "champion_test": final_test_table.iloc[0].to_dict(),
        "champion_period_table": champion_period_table,
        "champion_predictions": test_prediction,
        "validation_predictions_by_model": validation_predictions,
        "test_predictions_by_model": {champion_key: test_prediction},
        "train_only_models": model_bundles,
        "final_models": final_models,
        "best_params_by_model": {
            model_key: row["params_payload"] for model_key, row in best_rows_by_model.items()
        },
        "grid_search_tables": {},
        "grid_search_table": pd.DataFrame(search_rows),
        "prepared_data": prepared,
        "importance_tables": _build_importance_tables(
            final_models=final_models,
            feature_columns=prepared["feature_columns"],
        ),
        "selected_model_key": champion_key,
    }

    search_display = pd.DataFrame(
        [
            {
                key: value
                for key, value in row.items()
                if key
                not in {
                    "params_payload",
                    "validation_metrics",
                    "validation_metrics_12m",
                    "validation_metrics_lifetime",
                }
            }
            for row in search_rows
        ]
    )
    search_display["search_rank"] = range(1, len(search_display) + 1)
    result["grid_search_table"] = search_display.copy()
    search_display.to_csv(SEARCH_TABLE_PATH, index=False)
    pd.concat(
        [
            validation_predictions["random_forest"][_prediction_export_columns(validation_predictions["random_forest"])].assign(model_key="random_forest"),
            validation_predictions["xgboost"][_prediction_export_columns(validation_predictions["xgboost"])].assign(model_key="xgboost"),
        ],
        ignore_index=True,
    ).to_csv(VALIDATION_BEST_PATH, index=False)

    runtime = {
        "total_seconds": float(time.perf_counter() - start_total),
        "candidate_count_planned": len(candidates),
        "candidate_count_ran": len(search_rows),
        "candidate_runtime_rows": runtime_rows,
        "final_champion_fit_seconds": final_fit_seconds,
        "final_test_score_seconds": test_score_seconds,
        "selected_model_key": champion_key,
        "selected_model_name": MODEL_LABELS[champion_key],
        "search_table_path": str(SEARCH_TABLE_PATH),
        "view_path": str(VIEW_PATH),
        "prediction_path": config["output_path"],
    }
    RUNTIME_PATH.write_text(
        json.dumps(runtime, indent=2, default=_json_default),
        encoding="utf-8",
    )

    stage2_config = _write_stage2_config(config, result, view_summary, runtime)

    print("Exporting temporal report and figures...", flush=True)
    temporal_visuals = export_temporal_visuals(
        result,
        output_dir=REPORT_DIR / "figures" / "temporal_model",
        feature_search_result=None,
        show=False,
    )
    report_path = write_temporal_report(
        result,
        config,
        config["report_output_path"],
        feature_search_result=None,
        visual_paths=temporal_visuals,
    )
    _append_30min_report_note(Path(report_path), result, runtime)

    print("Running loss reserve workflow with 30-minute champion config...", flush=True)
    loss_config = build_loss_config(stage2_config)
    loss_result = run_loss_reserve_workflow(loss_config)
    loss_visuals = export_loss_reserve_visuals(
        loss_result,
        output_dir=REPORT_DIR / "figures" / "loss_reserve",
        show=False,
    )

    print(
        json.dumps(
            {
                "selected_model": champion_key,
                "champion_validation": {
                    "validation_auc": result["champion_validation"]["validation_auc"],
                    "validation_12m_auc": result["champion_validation"]["validation_12m_auc"],
                    "validation_lifetime_auc": result["champion_validation"]["validation_lifetime_auc"],
                },
                "champion_test": {
                    "test_auc": result["champion_test"]["test_auc"],
                    "test_12m_auc": result["champion_test"]["test_12m_auc"],
                    "test_lifetime_auc": result["champion_test"]["test_lifetime_auc"],
                },
                "runtime_seconds": runtime["total_seconds"],
                "pd_predictions_path": config["output_path"],
                "stage2_config_path": config["stage2_champion_config_path"],
                "temporal_report_path": str(report_path),
                "loss_report_path": str(loss_result["report_path"]),
                "loss_visual_count": len(loss_visuals.get("paths", loss_visuals)),
            },
            indent=2,
            default=_json_default,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
