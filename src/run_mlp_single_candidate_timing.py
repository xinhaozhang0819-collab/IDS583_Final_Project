from __future__ import annotations

import json
import time
from pathlib import Path

from calendar_bigdata import read_modeling_sample
from experiment import (
    FULL_CALENDAR_HAZARD_PROFILE,
    PD_MODELING_TYPE_CALENDAR_TIME_HAZARD,
    TARGET_1M_COLUMN,
    _build_calendar_hazard_panel_payload,
    _build_hazard_design_matrix,
    _fit_hazard_model_from_payload,
    _safe_evaluate_model,
    prepare_experiment_data,
)
from run_hazard_pd_pipeline import PROCESSED_DIR, PROJECT_ROOT, build_pd_config


OUTPUT_DIR = PROCESSED_DIR / "mlp_single_candidate_timing"
OUTPUT_PATH = OUTPUT_DIR / "metrics.json"


def _json_default(value):
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def main() -> None:
    start_total = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = build_pd_config()
    config.update(
        {
            "enabled_models": ["mlp_neural_network"],
            "grid_search_models": ["mlp_neural_network"],
            "use_grid_search": False,
            "calendar_panel_is_prebuilt": True,
            "calendar_panel_path": None,
            "calendar_panel_counts_path": None,
        }
    )
    mlp_params = {
        "hidden_layer_sizes": (64, 32, 16),
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.0001,
        "learning_rate_init": 0.001,
        "max_iter": 30,
        "batch_size": 4096,
        "early_stopping": True,
        "validation_fraction": 0.05,
        "n_iter_no_change": 3,
        "random_state": 42,
    }
    config["model_params"]["mlp_neural_network"].update(mlp_params)

    modeling_sample_path = PROCESSED_DIR / "calendar_survival_modeling_sample.parquet"
    print(f"Reading modeling sample: {modeling_sample_path}", flush=True)
    read_start = time.perf_counter()
    modeling_sample = read_modeling_sample(modeling_sample_path)
    read_seconds = time.perf_counter() - read_start
    print(f"Rows: {len(modeling_sample):,}; read_seconds={read_seconds:.1f}", flush=True)

    prepared = prepare_experiment_data(config, modeling_sample)
    split_summary = prepared["split_summary"].copy()
    print(split_summary.to_string(index=False), flush=True)

    payload_start = time.perf_counter()
    train_payload = _build_calendar_hazard_panel_payload(
        prepared["train"]["df"],
        feature_profile=FULL_CALENDAR_HAZARD_PROFILE,
        meta_columns=prepared["meta_columns"],
        mlp_sampling_config=config["mlp_sampling"],
        model_name="mlp_neural_network",
    )
    payload_seconds = time.perf_counter() - payload_start
    print(
        f"Train rows after MLP sampling: {len(train_payload['df']):,}; "
        f"features={len(train_payload['feature_columns'])}; "
        f"payload_seconds={payload_seconds:.1f}",
        flush=True,
    )
    print(f"MLP params: {json.dumps(mlp_params, default=_json_default)}", flush=True)

    fit_start = time.perf_counter()
    model_bundle = _fit_hazard_model_from_payload(
        "mlp_neural_network",
        mlp_params,
        train_payload,
        prepared["meta_columns"],
        feature_mode="calendar_time",
    )
    fit_seconds = time.perf_counter() - fit_start
    model_bundle["modeling_type"] = PD_MODELING_TYPE_CALENDAR_TIME_HAZARD
    print(f"fit_seconds={fit_seconds:.1f}", flush=True)
    fit_only_result = {
        "experiment": "mlp_single_candidate_timing",
        "phase": "fit_complete",
        "train_rows_after_mlp_sampling": int(len(train_payload["df"])),
        "feature_count": int(len(train_payload["feature_columns"])),
        "params": mlp_params,
        "fit_seconds": float(fit_seconds),
        "n_iter_": int(getattr(model_bundle["model"].named_steps["classifier"], "n_iter_", -1)),
    }
    (OUTPUT_DIR / "fit_timing.json").write_text(
        json.dumps(fit_only_result, indent=2, default=_json_default),
        encoding="utf-8",
    )

    score_start = time.perf_counter()
    validation_design = _build_hazard_design_matrix(
        prepared["validation"]["df"],
        feature_profile=model_bundle["feature_profile"],
        design_columns=model_bundle["feature_columns"],
    )
    predicted_hazard_1m = model_bundle["model"].predict_proba(validation_design)[:, 1].clip(0, 1)
    validation_score_seconds = time.perf_counter() - score_start
    print(f"validation_1m_score_seconds={validation_score_seconds:.1f}", flush=True)

    metrics_1m = _safe_evaluate_model(
        prepared["validation"]["df"][TARGET_1M_COLUMN],
        predicted_hazard_1m,
    )
    classifier = model_bundle["model"].named_steps["classifier"]
    result = {
        "experiment": "mlp_single_candidate_timing",
        "project_root": str(PROJECT_ROOT),
        "modeling_sample_path": str(modeling_sample_path),
        "modeling_sample_rows": int(len(modeling_sample)),
        "split_summary": split_summary.to_dict(orient="records"),
        "train_rows_after_mlp_sampling": int(len(train_payload["df"])),
        "feature_count": int(len(train_payload["feature_columns"])),
        "params": mlp_params,
        "read_seconds": float(read_seconds),
        "payload_seconds": float(payload_seconds),
        "fit_seconds": float(fit_seconds),
        "validation_1m_score_seconds": float(validation_score_seconds),
        "total_seconds": float(time.perf_counter() - start_total),
        "n_iter_": int(getattr(classifier, "n_iter_", -1)),
        "loss_": float(getattr(classifier, "loss_", float("nan"))),
        "best_validation_score_": float(
            getattr(classifier, "best_validation_score_", float("nan"))
        ),
        "validation_metrics_1m": metrics_1m,
        "prediction_checks": {
            "hazard_min": float(predicted_hazard_1m.min()),
            "hazard_max": float(predicted_hazard_1m.max()),
        },
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps(result, indent=2, default=_json_default), flush=True)


if __name__ == "__main__":
    main()
