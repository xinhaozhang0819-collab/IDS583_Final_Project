from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import pandas as pd

from calendar_bigdata import read_modeling_sample
from experiment import (
    FULL_CALENDAR_HAZARD_PROFILE,
    PD_MODELING_TYPE_CALENDAR_TIME_HAZARD,
    TARGET_1M_COLUMN,
    _build_calendar_hazard_panel_payload,
    _fit_hazard_model_from_payload,
    _metric_comparison_key,
    _safe_evaluate_model,
    _score_calendar_time_hazard_model_bundle,
    prepare_experiment_data,
    rank_validation_results,
)
from run_hazard_pd_pipeline import PROCESSED_DIR, PROJECT_ROOT, REPORT_DIR, build_pd_config


EXPERIMENT_NAME = "mlp_1p77m_6candidate_3layer"
OUTPUT_DIR = PROCESSED_DIR / EXPERIMENT_NAME
REPORT_PATH = REPORT_DIR / f"{EXPERIMENT_NAME}_report.md"


THREE_LAYER_CANDIDATES = [
    {
        "hidden_layer_sizes": (64, 32, 16),
        "alpha": 0.0001,
        "learning_rate_init": 0.001,
    },
    {
        "hidden_layer_sizes": (64, 32, 16),
        "alpha": 0.001,
        "learning_rate_init": 0.001,
    },
    {
        "hidden_layer_sizes": (96, 48, 24),
        "alpha": 0.0001,
        "learning_rate_init": 0.001,
    },
    {
        "hidden_layer_sizes": (96, 48, 24),
        "alpha": 0.001,
        "learning_rate_init": 0.001,
    },
    {
        "hidden_layer_sizes": (128, 64, 32),
        "alpha": 0.0001,
        "learning_rate_init": 0.001,
    },
    {
        "hidden_layer_sizes": (128, 64, 32),
        "alpha": 0.001,
        "learning_rate_init": 0.001,
    },
]


def _json_default(value):
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _params_payload(base_params: dict, candidate_params: dict) -> dict:
    params = copy.deepcopy(base_params)
    params.update(candidate_params)
    return params


def _metric_row(candidate_id: int, params: dict, prediction: pd.DataFrame, elapsed: float) -> dict:
    metrics_1m = _safe_evaluate_model(
        prediction[TARGET_1M_COLUMN],
        prediction["predicted_hazard_1m"],
    )
    metrics_12m = _safe_evaluate_model(
        prediction["actual_default_12m"],
        prediction["predicted_pd_12m"],
    )
    return {
        "candidate_id": candidate_id,
        "model_key": "mlp_neural_network",
        "model_name": "MLP Neural Network",
        "params": json.dumps(params, sort_keys=True, default=_json_default),
        "params_payload": copy.deepcopy(params),
        "validation_auc": metrics_1m["AUC"],
        "validation_ks": metrics_1m["KS"],
        "validation_brier": metrics_1m["Brier"],
        "validation_12m_auc": metrics_12m["AUC"],
        "validation_12m_brier": metrics_12m["Brier"],
        "calibration_summary": metrics_1m["calibration_summary"],
        "validation_metrics": metrics_1m,
        "validation_metrics_12m": metrics_12m,
        "fit_score_seconds": elapsed,
    }


def _write_search_outputs(rows: list[dict]) -> pd.DataFrame:
    display_rows = []
    for row in rows:
        display = {
            key: value
            for key, value in row.items()
            if key not in {"params_payload", "validation_metrics", "validation_metrics_12m"}
        }
        display_rows.append(display)
    search_table = rank_validation_results(pd.DataFrame(display_rows))
    if not search_table.empty:
        search_table["search_rank"] = range(1, len(search_table) + 1)
    search_table.to_csv(OUTPUT_DIR / "search_table.csv", index=False)
    (OUTPUT_DIR / "search_rows.json").write_text(
        json.dumps(rows, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return search_table


def _best_row(rows: list[dict]) -> dict:
    return max(rows, key=lambda row: _metric_comparison_key(row["validation_metrics"]))


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
            "actual_default_12m",
            "predicted_hazard_1m",
            "predicted_pd_12m",
            "predicted_pd",
            "model_name",
        ]
        if column in prediction.columns
    ]


def _write_report(summary: dict, search_table: pd.DataFrame) -> None:
    lines = [
        "# MLP 1.77M Calendar-Time Hazard Experiment",
        "",
        "## Setup",
        "",
        f"- Modeling sample path: `{summary['modeling_sample_path']}`",
        f"- Modeling sample rows: `{summary['modeling_sample_rows']}`",
        f"- MLP train rows after internal sampling: `{summary['mlp_train_rows']}`",
        f"- Candidate count: `{summary['candidate_count']}`",
        f"- Hidden layer policy: all candidates use three hidden layers.",
        f"- Best candidate id: `{summary['best_candidate_id']}`",
        f"- Best params: `{json.dumps(summary['best_params'], default=_json_default)}`",
        f"- Total seconds: `{summary['total_seconds']:.1f}`",
        "",
        "## Validation Search Results",
        "",
        search_table.to_markdown(index=False),
        "",
        "## Final Test Results",
        "",
        f"- Test AUC: `{summary['test_metrics']['AUC']:.4f}`",
        f"- Test KS: `{summary['test_metrics']['KS']:.4f}`",
        f"- Test Brier: `{summary['test_metrics']['Brier']:.4f}`",
        f"- Test 12m AUC: `{summary['test_metrics_12m']['AUC']:.4f}`",
        f"- Test 12m Brier: `{summary['test_metrics_12m']['Brier']:.4f}`",
        "",
        "## Output Files",
        "",
        f"- Search table: `{OUTPUT_DIR / 'search_table.csv'}`",
        f"- Validation predictions: `{OUTPUT_DIR / 'best_validation_predictions.csv'}`",
        f"- Test predictions: `{OUTPUT_DIR / 'best_test_predictions.csv'}`",
        f"- Metrics JSON: `{OUTPUT_DIR / 'metrics.json'}`",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    start_total = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    config = build_pd_config()
    config.update(
        {
            "enabled_models": ["mlp_neural_network"],
            "grid_search_models": ["mlp_neural_network"],
            "use_grid_search": True,
            "calendar_panel_is_prebuilt": True,
            "calendar_panel_path": None,
            "calendar_panel_counts_path": None,
            "output_path": str(OUTPUT_DIR / "best_test_predictions.csv"),
            "stage2_champion_config_path": str(OUTPUT_DIR / "stage2_champion_config.json"),
            "report_output_path": str(REPORT_PATH),
        }
    )
    config["model_params"]["mlp_neural_network"].update(
        {
            "hidden_layer_sizes": (64, 32, 16),
            "alpha": 0.0001,
            "learning_rate_init": 0.001,
        }
    )
    config["param_grid"] = {
        "mlp_neural_network": {
            "hidden_layer_sizes": [candidate["hidden_layer_sizes"] for candidate in THREE_LAYER_CANDIDATES[::2]],
            "alpha": [0.0001, 0.001],
        }
    }

    modeling_sample_path = PROCESSED_DIR / "calendar_survival_modeling_sample.parquet"
    print(f"Reading modeling sample: {modeling_sample_path}", flush=True)
    modeling_sample = read_modeling_sample(modeling_sample_path)
    print(f"Rows: {len(modeling_sample):,}", flush=True)

    prepared = prepare_experiment_data(config, modeling_sample)
    split_summary = prepared["split_summary"].copy()
    split_summary.to_csv(OUTPUT_DIR / "split_summary.csv", index=False)
    print(split_summary.to_string(index=False), flush=True)

    train_payload = _build_calendar_hazard_panel_payload(
        prepared["train"]["df"],
        feature_profile=FULL_CALENDAR_HAZARD_PROFILE,
        meta_columns=prepared["meta_columns"],
        mlp_sampling_config=config["mlp_sampling"],
        model_name="mlp_neural_network",
    )
    print(
        f"MLP train rows after internal sampling: {len(train_payload['df']):,}; "
        f"features={len(train_payload['feature_columns'])}",
        flush=True,
    )

    base_params = copy.deepcopy(config["model_params"]["mlp_neural_network"])
    rows = []
    best_prediction = None
    best_params = None
    best_key = None
    for candidate_id, candidate in enumerate(THREE_LAYER_CANDIDATES, start=1):
        params = _params_payload(base_params, candidate)
        print(
            f"Candidate {candidate_id}/6 params={json.dumps(params, default=_json_default)}",
            flush=True,
        )
        start = time.perf_counter()
        model_bundle = _fit_hazard_model_from_payload(
            "mlp_neural_network",
            params,
            train_payload,
            prepared["meta_columns"],
            feature_mode="calendar_time",
        )
        model_bundle["modeling_type"] = PD_MODELING_TYPE_CALENDAR_TIME_HAZARD
        validation_prediction = _score_calendar_time_hazard_model_bundle(
            model_bundle,
            prepared["validation"]["df"],
        )
        elapsed = time.perf_counter() - start
        row = _metric_row(candidate_id, params, validation_prediction, elapsed)
        rows.append(row)
        search_table = _write_search_outputs(rows)
        comparison_key = _metric_comparison_key(row["validation_metrics"])
        if best_key is None or comparison_key > best_key:
            best_key = comparison_key
            best_prediction = validation_prediction
            best_params = params
            best_prediction[_prediction_export_columns(best_prediction)].to_csv(
                OUTPUT_DIR / "best_validation_predictions.csv",
                index=False,
            )
        print(
            f"Candidate {candidate_id} done in {elapsed:.1f}s; "
            f"validation_auc={row['validation_auc']:.4f}; "
            f"validation_ks={row['validation_ks']:.4f}; "
            f"validation_12m_auc={row['validation_12m_auc']:.4f}",
            flush=True,
        )

    best = _best_row(rows)
    best_params = copy.deepcopy(best["params_payload"])
    print(f"Best candidate: {best['candidate_id']} params={best['params']}", flush=True)
    print("Refitting best MLP on train + validation...", flush=True)
    train_validation_panel = pd.concat(
        [prepared["train"]["df"], prepared["validation"]["df"]],
        ignore_index=True,
    )
    final_payload = _build_calendar_hazard_panel_payload(
        train_validation_panel,
        feature_profile=FULL_CALENDAR_HAZARD_PROFILE,
        meta_columns=prepared["meta_columns"],
        mlp_sampling_config=config["mlp_sampling"],
        model_name="mlp_neural_network",
    )
    final_model = _fit_hazard_model_from_payload(
        "mlp_neural_network",
        best_params,
        final_payload,
        prepared["meta_columns"],
        feature_mode="calendar_time",
    )
    final_model["modeling_type"] = PD_MODELING_TYPE_CALENDAR_TIME_HAZARD
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
    test_prediction[_prediction_export_columns(test_prediction)].to_csv(
        OUTPUT_DIR / "best_test_predictions.csv",
        index=False,
    )

    search_table = _write_search_outputs(rows)
    summary = {
        "experiment_name": EXPERIMENT_NAME,
        "project_root": str(PROJECT_ROOT),
        "modeling_sample_path": str(modeling_sample_path),
        "modeling_sample_rows": int(len(modeling_sample)),
        "split_summary": split_summary.to_dict(orient="records"),
        "mlp_train_rows": int(len(train_payload["df"])),
        "mlp_final_train_validation_rows": int(len(final_payload["df"])),
        "feature_count": int(len(train_payload["feature_columns"])),
        "candidate_count": len(THREE_LAYER_CANDIDATES),
        "best_candidate_id": int(best["candidate_id"]),
        "best_params": best_params,
        "best_validation": {
            "validation_auc": float(best["validation_auc"]),
            "validation_ks": float(best["validation_ks"]),
            "validation_brier": float(best["validation_brier"]),
            "validation_12m_auc": float(best["validation_12m_auc"]),
            "validation_12m_brier": float(best["validation_12m_brier"]),
        },
        "test_metrics": test_metrics,
        "test_metrics_12m": test_metrics_12m,
        "total_seconds": float(time.perf_counter() - start_total),
    }
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(summary, indent=2, default=_json_default),
        encoding="utf-8",
    )
    _write_report(summary, search_table)
    print(json.dumps(summary, indent=2, default=_json_default), flush=True)


if __name__ == "__main__":
    main()
