from __future__ import annotations

import copy
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from calendar_bigdata import read_modeling_sample
from experiment import fit_locked_model
from loss_preprocess import load_loss_raw_dataset
from loss_workflow import _score_stage2_subset, run_loss_reserve_workflow
from run_hazard_pd_pipeline import PROCESSED_DIR, REPORT_DIR, build_loss_config
from visualization import export_loss_reserve_visuals


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE2_CONFIG_PATH = PROCESSED_DIR / "stage2_champion_config.json"
VALIDATION_PREDICTION_PATH = PROCESSED_DIR / "calendar_30min_model_best_validation.csv"
PANEL_SUMMARY_PATH = PROCESSED_DIR / "calendar_survival_panel_summary.json"
TIME_ESTIMATE_PATH = PROCESSED_DIR / "stage2_full_scoring_time_estimate.json"

BENCHMARK_ROWS_PER_SCOPE = 10_000
MAX_ESTIMATED_SECONDS = 3 * 60 * 60
MAX_MEMORY_SHARE = 0.70


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _weighted_mean(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    return float(np.sum(values * weights) / np.sum(weights))


def _apply_logit_shift(probabilities, shift, eps=1e-6):
    p = np.clip(np.asarray(probabilities, dtype=float), eps, 1.0 - eps)
    logits = np.log(p / (1.0 - p)) + shift
    logits = np.clip(logits, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-logits))


def _fit_intercept_shift(probabilities, weights, target_mean):
    low, high = -50.0, 50.0
    for _ in range(100):
        mid = (low + high) / 2.0
        shifted_mean = _weighted_mean(_apply_logit_shift(probabilities, mid), weights)
        if shifted_mean < target_mean:
            low = mid
        else:
            high = mid
    return float((low + high) / 2.0)


def _build_validation_weights(validation_df, panel_summary):
    y = validation_df["actual_default"].fillna(0).astype(int).to_numpy()
    full_events = float(panel_summary["event_counts"]["validation"])
    full_rows = float(panel_summary["split_counts"]["validation"])
    sample_events = max(float(y.sum()), 1.0)
    sample_non_events = max(float(len(y) - y.sum()), 1.0)
    weights = np.where(
        y == 1,
        full_events / sample_events,
        (full_rows - full_events) / sample_non_events,
    )
    return y, weights, full_events / full_rows


def fit_hazard_calibration():
    validation = pd.read_csv(VALIDATION_PREDICTION_PATH)
    validation = validation[validation["model_key"].eq("xgboost")].copy()
    panel_summary = _load_json(PANEL_SUMMARY_PATH)
    y, weights, target_mean = _build_validation_weights(validation, panel_summary)
    raw = validation["predicted_hazard_1m"].astype(float).clip(0, 1).to_numpy()
    shift = _fit_intercept_shift(raw, weights, target_mean)
    calibrated = _apply_logit_shift(raw, shift)
    raw_brier = float(np.average((raw - y) ** 2, weights=weights))
    calibrated_brier = float(np.average((calibrated - y) ** 2, weights=weights))
    return {
        "method": "logit_intercept",
        "intercept_shift": shift,
        "epsilon": 1e-6,
        "validation_full_event_rate": float(target_mean),
        "validation_sample_rows": int(len(validation)),
        "validation_sample_event_rate": float(y.mean()),
        "validation_sample_raw_mean_hazard": float(raw.mean()),
        "validation_weighted_raw_mean_hazard": _weighted_mean(raw, weights),
        "validation_weighted_calibrated_mean_hazard": _weighted_mean(calibrated, weights),
        "validation_weighted_raw_brier": raw_brier,
        "validation_weighted_calibrated_brier": calibrated_brier,
        "validation_calibrated_brier_not_higher": bool(calibrated_brier <= raw_brier),
    }


def _memory_snapshot():
    try:
        import psutil
    except ImportError:
        return {"rss_bytes": None, "available_bytes": None, "rss_share_of_available": None}
    process = psutil.Process()
    available = psutil.virtual_memory().available
    rss = process.memory_info().rss
    return {
        "rss_bytes": int(rss),
        "available_bytes": int(available),
        "rss_share_of_available": float(rss / available) if available else None,
    }


def _sample_ids(path, n, random_state):
    df = pd.read_csv(path, usecols=["sample_id"])
    ids = df["sample_id"].dropna().drop_duplicates()
    if len(ids) <= n:
        return ids
    return ids.sample(n=n, random_state=random_state)


def benchmark_stage2_scoring(stage2_config):
    raw_loss_df = load_loss_raw_dataset(stage2_config["processed_data_path"])
    if "sample_id" not in raw_loss_df.columns:
        raw_loss_df["sample_id"] = np.arange(len(raw_loss_df))

    training_df = read_modeling_sample(stage2_config["calendar_modeling_sample_path"])
    bundle = fit_locked_model(
        stage2_config,
        training_df,
        model_name=stage2_config.get("selected_model_key"),
    )

    resolved_ids = _sample_ids(
        PROCESSED_DIR / "test_with_loss_metrics.csv",
        BENCHMARK_ROWS_PER_SCOPE,
        int(stage2_config.get("random_state", 42)),
    )
    active_ids = _sample_ids(
        PROCESSED_DIR / "cecl_active_snapshot.csv",
        BENCHMARK_ROWS_PER_SCOPE,
        int(stage2_config.get("random_state", 42)) + 1,
    )

    before = _memory_snapshot()
    start = time.perf_counter()
    resolved_prediction = _score_stage2_subset(bundle, raw_loss_df, resolved_ids)
    active_prediction = _score_stage2_subset(bundle, raw_loss_df, active_ids)
    sequential_seconds = time.perf_counter() - start
    after_sequential = _memory_snapshot()

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as executor:
        resolved_future = executor.submit(_score_stage2_subset, bundle, raw_loss_df, resolved_ids)
        active_future = executor.submit(_score_stage2_subset, bundle, raw_loss_df, active_ids)
        resolved_parallel = resolved_future.result()
        active_parallel = active_future.result()
    parallel_seconds = time.perf_counter() - start
    after_parallel = _memory_snapshot()

    _assert_prediction_bounds(resolved_prediction, "resolved benchmark")
    _assert_prediction_bounds(active_prediction, "active benchmark")
    _assert_prediction_bounds(resolved_parallel, "resolved parallel benchmark")
    _assert_prediction_bounds(active_parallel, "active parallel benchmark")

    total_rows = 225_611 + 912_609
    benchmark_rows = int(resolved_prediction["sample_id"].nunique() + active_prediction["sample_id"].nunique())
    sequential_estimate = sequential_seconds / benchmark_rows * total_rows
    parallel_estimate = parallel_seconds / benchmark_rows * total_rows
    use_parallel = parallel_estimate < sequential_estimate
    selected_estimate = parallel_estimate if use_parallel else sequential_estimate
    selected_parallel_jobs = 2 if use_parallel else 1
    memory_snapshot = after_parallel if use_parallel else after_sequential

    return {
        "benchmark_rows_per_scope": BENCHMARK_ROWS_PER_SCOPE,
        "resolved_benchmark_rows": int(resolved_prediction["sample_id"].nunique()),
        "active_benchmark_rows": int(active_prediction["sample_id"].nunique()),
        "sequential_benchmark_seconds": float(sequential_seconds),
        "parallel_benchmark_seconds": float(parallel_seconds),
        "estimated_full_sequential_seconds": float(sequential_estimate),
        "estimated_full_parallel_seconds": float(parallel_estimate),
        "selected_parallel_jobs": selected_parallel_jobs,
        "estimated_full_seconds": float(selected_estimate),
        "estimated_full_minutes": float(selected_estimate / 60.0),
        "memory_before": before,
        "memory_after_selected_mode": memory_snapshot,
        "passes_time_budget": bool(selected_estimate <= MAX_ESTIMATED_SECONDS),
        "passes_memory_budget": bool(
            memory_snapshot.get("rss_share_of_available") is None
            or memory_snapshot["rss_share_of_available"] <= MAX_MEMORY_SHARE
        ),
    }


def _assert_prediction_bounds(prediction, label):
    for column in ["predicted_pd", "predicted_pd_12m", "predicted_pd_lifetime"]:
        if column in prediction.columns and not prediction[column].between(0, 1).all():
            raise ValueError(f"{label} {column} outside [0, 1].")
    if {"predicted_pd_12m", "predicted_pd_lifetime"}.issubset(prediction.columns):
        if not (prediction["predicted_pd_12m"] <= prediction["predicted_pd_lifetime"]).all():
            raise ValueError(f"{label} has 12m PD greater than lifetime PD.")


def _prepare_stage2_config():
    stage2_config = _load_json(STAGE2_CONFIG_PATH)
    stage2_config = copy.deepcopy(stage2_config)
    stage2_config["full_stage2_scoring"] = True
    stage2_config["max_stage2_scoring_rows_per_scope"] = None
    stage2_config["stage2_scoring_chunk_rows"] = int(
        stage2_config.get("stage2_scoring_chunk_rows", 4000) or 4000
    )
    stage2_config["stage2_time_estimate_path"] = str(TIME_ESTIMATE_PATH)
    model_params = stage2_config.setdefault("model_params", {}).setdefault("xgboost", {})
    model_params["n_jobs"] = max(1, (os.cpu_count() or 2) // 2)
    stage2_config["hazard_calibration"] = fit_hazard_calibration()
    return stage2_config


def main():
    stage2_config = _prepare_stage2_config()
    benchmark = benchmark_stage2_scoring(stage2_config)
    stage2_config["stage2_scoring_parallel_jobs"] = benchmark["selected_parallel_jobs"]
    estimate_payload = {
        "stage": "pre_full_run_estimate",
        "stage2_config_path": str(STAGE2_CONFIG_PATH),
        "hazard_calibration": stage2_config["hazard_calibration"],
        **benchmark,
    }
    _write_json(TIME_ESTIMATE_PATH, estimate_payload)
    _write_json(STAGE2_CONFIG_PATH, stage2_config)

    if not benchmark["passes_time_budget"] or not benchmark["passes_memory_budget"]:
        print(json.dumps(estimate_payload, indent=2, default=_json_default))
        raise SystemExit("Full stage2 scoring estimate exceeded time or memory budget.")

    loss_config = build_loss_config(stage2_config)
    result = run_loss_reserve_workflow(loss_config)
    visuals = export_loss_reserve_visuals(
        result,
        output_dir=REPORT_DIR / "figures" / "loss_reserve",
        show=False,
    )
    visual_paths = visuals.get("paths", visuals) if isinstance(visuals, dict) else visuals
    refreshed = {
        "stage": "full_run_complete",
        "loss_report_path": str(result["report_path"]),
        "portfolio_summary_path": loss_config["portfolio_summary_path"],
        "segment_summary_path": loss_config["segment_summary_path"],
        "visual_count": len(visual_paths),
        "stage2_config_path": str(STAGE2_CONFIG_PATH),
        **benchmark,
    }
    _write_json(TIME_ESTIMATE_PATH, refreshed)
    print(json.dumps(refreshed, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
