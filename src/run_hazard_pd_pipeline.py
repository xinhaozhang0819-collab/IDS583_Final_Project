from __future__ import annotations

import copy
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from calendar_bigdata import (
    build_calendar_panel_with_dask,
    build_calendar_sample_only_with_dask,
    read_modeling_sample,
)
from experiment import _parameter_candidates, _search_calendar_time_hazard_model, prepare_experiment_data, run_experiment
from loss_preprocess import load_loss_raw_dataset
from loss_workflow import run_loss_reserve_workflow
from reporting import write_temporal_report
from visualization import export_loss_reserve_visuals, export_temporal_visuals


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
REPORT_DIR = PROJECT_ROOT / "reports"


def build_pd_config() -> dict:
    return {
        "modeling_type": "calendar_time_hazard",
        "hazard_feature_mode": "calendar_time",
        "data_cutoff": "2018-12-01",
        "calendar_panel_path": str(PROCESSED_DIR / "calendar_survival_panel.parquet"),
        "calendar_panel_counts_path": str(PROCESSED_DIR / "calendar_survival_panel_counts.parquet"),
        "calendar_panel_summary_path": str(PROCESSED_DIR / "calendar_survival_panel_summary.json"),
        "calendar_modeling_sample_path": str(PROCESSED_DIR / "calendar_survival_modeling_sample.parquet"),
        "calendar_modeling_sample_summary_path": str(PROCESSED_DIR / "calendar_survival_modeling_sample_summary.json"),
        "calendar_training_time_estimate_path": str(PROCESSED_DIR / "calendar_training_time_estimate.json"),
        "calendar_resource_bounded_sample_path": str(
            PROCESSED_DIR / "calendar_survival_modeling_sample_resource_bounded.parquet"
        ),
        "calendar_resource_bounded_sample_summary_path": str(
            PROCESSED_DIR / "calendar_survival_modeling_sample_resource_bounded_summary.json"
        ),
        "calendar_bigdata_backend": "dask",
        "calendar_bigdata_mode": "sample_only",
        "reuse_calendar_bigdata_outputs": True,
        "run_resource_bounded_after_estimate_exceeds_threshold": True,
        "calendar_bigdata_blocksize": "8MB",
        "calendar_bigdata_sample": {
            "negative_to_positive_ratio": 10,
            "max_train_rows": 750_000,
            "max_validation_rows": 500_000,
            "max_test_rows": 500_000,
            "include_scoring_rows": True,
            "max_scoring_rows": 100_000,
            "random_state": 42,
        },
        "calendar_training_time_estimate": {
            "max_train_rows": 10_000,
            "max_validation_rows": 5_000,
            "max_test_rows": 2_000,
            "negative_to_positive_ratio": 5,
            "row_scaling_power": 1.0,
            "safety_factor": 2.0,
            "threshold_hours": 8.0,
        },
        "resource_bounded_training_sample": {
            "max_train_rows": 5_000,
            "max_validation_rows": 2_500,
            "max_test_rows": 10_000,
            "negative_to_positive_ratio": 5,
            "random_state": 42,
        },
        "processed_data_path": str(DATA_DIR / "raw" / "accepted_2007_to_2018Q4.csv"),
        "output_path": str(PROCESSED_DIR / "test_with_pd_best_model.csv"),
        "stage2_champion_config_path": str(PROCESSED_DIR / "stage2_champion_config.json"),
        "report_output_path": str(REPORT_DIR / "temporal_model_report.md"),
        "random_state": 42,
        "sample_frac": 1.0,
        "generate_full_report": True,
        "use_grid_search": True,
        "grid_search_models": [
            "logistic_regression",
            "random_forest",
            "xgboost",
            "mlp_neural_network",
        ],
        "temporal_split": {
            "train_end": "2016-01-01",
            "valid_end": "2017-01-01",
        },
        "enabled_models": [
            "logistic_regression",
            "random_forest",
            "xgboost",
            "mlp_neural_network",
        ],
        "meta_columns": [
            "sample_id",
            "snapshot_month",
            "issue_date",
            "loan_amnt",
            "annual_inc",
            "fico_range_low",
            "term_months",
            "int_rate",
            "revol_util",
            "month_on_book",
            "remaining_term",
            "target_1m",
        ],
        "mlp_sampling": {
            "negative_to_positive_ratio": 10,
            "max_train_rows": 750000,
            "random_state": 42,
        },
        "model_params": {
            "logistic_regression": {
                "C": 1.0,
                "max_iter": 1000,
                "class_weight": None,
            },
            "random_forest": {
                "n_estimators": 120,
                "max_depth": 14,
                "min_samples_split": 100,
                "min_samples_leaf": 25,
                "max_features": "sqrt",
                "n_jobs": -1,
                "random_state": 42,
            },
            "xgboost": {
                "n_estimators": 120,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 5,
                "reg_lambda": 1.0,
                "random_state": 42,
                "n_jobs": -1,
            },
            "mlp_neural_network": {
                "activation": "relu",
                "solver": "adam",
                "max_iter": 120,
                "early_stopping": True,
                "validation_fraction": 0.1,
                "n_iter_no_change": 10,
                "random_state": 42,
            },
        },
        "param_grid": {
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
        },
        "run_feature_search": False,
    }


def build_loss_config(stage2_config: dict) -> dict:
    return {
        "raw_data_path": str(DATA_DIR / "raw" / "accepted_2007_to_2018Q4.csv"),
        "pd_prediction_path": str(PROCESSED_DIR / "test_with_pd_best_model.csv"),
        "loss_workflow_path": str(PROCESSED_DIR / "loss_workflow_dataset.csv"),
        "charged_off_proxy_path": str(PROCESSED_DIR / "charged_off_loss_proxy.csv"),
        "cecl_snapshot_path": str(PROCESSED_DIR / "cecl_active_snapshot.csv"),
        "test_loss_metrics_path": str(PROCESSED_DIR / "test_with_loss_metrics.csv"),
        "segment_summary_path": str(PROCESSED_DIR / "segment_expected_loss_summary.csv"),
        "portfolio_summary_path": str(PROCESSED_DIR / "portfolio_expected_loss_summary.csv"),
        "hazard_comparison_path": str(PROCESSED_DIR / "hazard_model_comparison.csv"),
        "hazard_search_path": str(PROCESSED_DIR / "hazard_grid_search_results.csv"),
        "report_output_path": str(REPORT_DIR / "loss_reserve_segmentation_report.md"),
        "temporal_split": {
            "train_end": "2016-01-01",
            "valid_end": "2017-01-01",
        },
        "lgd_fit_splits": ["train", "validation"],
        "lgd_shrinkage_k": 50,
        "hazard_param_grid": {
            "C": [0.25, 0.5, 1.0, 2.0, 5.0],
            "class_weight": [None, "balanced"],
        },
        "hazard_ml_param_grid": {
            "learning_rate": [0.03, 0.05, 0.08],
            "max_iter": [120, 180],
            "max_depth": [4, 6],
            "min_samples_leaf": [80, 120],
            "l2_regularization": [0.0, 0.01],
        },
        "stage2_scoring": stage2_config,
    }


def estimate_calendar_training_time(config: dict, modeling_sample_df) -> dict:
    estimate_config = copy.deepcopy(config.get("calendar_training_time_estimate", {}))
    benchmark_df = _sample_training_time_benchmark_frame(
        modeling_sample_df,
        max_train_rows=int(estimate_config.get("max_train_rows", 100_000)),
        max_validation_rows=int(estimate_config.get("max_validation_rows", 50_000)),
        max_test_rows=int(estimate_config.get("max_test_rows", 10_000)),
        negative_to_positive_ratio=int(estimate_config.get("negative_to_positive_ratio", 5)),
        random_state=int(config.get("random_state", 42)),
    )
    benchmark_config = copy.deepcopy(config)
    benchmark_config["enabled_models"] = copy.deepcopy(config["enabled_models"])
    benchmark_config["use_grid_search"] = False
    benchmark_config["grid_search_models"] = []
    benchmark_config["calendar_panel_is_prebuilt"] = True

    prepared = prepare_experiment_data(benchmark_config, benchmark_df)
    model_rows = []
    total_estimated_seconds = 0.0
    row_scaling_power = float(estimate_config.get("row_scaling_power", 1.0))
    safety_factor = float(estimate_config.get("safety_factor", 1.5))
    threshold_hours = float(estimate_config.get("threshold_hours", 8.0))
    full_model_rows = int(
        len(
            modeling_sample_df[
                modeling_sample_df["split_label"].isin(["train", "validation"])
            ]
        )
    )
    benchmark_model_rows = max(
        1,
        int(
            len(
                benchmark_df[
                    benchmark_df["split_label"].isin(["train", "validation"])
                ]
            )
        ),
    )
    row_scale = (full_model_rows / benchmark_model_rows) ** row_scaling_power
    for model_name in benchmark_config["enabled_models"]:
        model_config = copy.deepcopy(benchmark_config)
        model_config["enabled_models"] = [model_name]
        candidates, _ = _parameter_candidates(model_name, config)
        print(
            f"Benchmarking {model_name} on {benchmark_model_rows:,} train+validation rows...",
            flush=True,
        )
        start = time.perf_counter()
        result = _search_calendar_time_hazard_model(
            model_name,
            model_config,
            prepared["train"],
            prepared["validation"],
        )
        elapsed = time.perf_counter() - start
        estimated_seconds = elapsed * len(candidates) * row_scale * safety_factor
        print(
            f"{model_name} base benchmark: {elapsed:.1f}s; "
            f"estimated grid: {estimated_seconds / 3600.0:.2f}h",
            flush=True,
        )
        total_estimated_seconds += estimated_seconds
        model_rows.append(
            {
                "model_key": model_name,
                "base_candidate_seconds": round(elapsed, 3),
                "benchmark_rows": int(len(benchmark_df)),
                "benchmark_model_rows": benchmark_model_rows,
                "full_model_rows": full_model_rows,
                "row_scale": round(row_scale, 3),
                "safety_factor": safety_factor,
                "grid_candidates": int(len(candidates)),
                "estimated_grid_seconds": round(estimated_seconds, 3),
                "estimated_grid_hours": round(estimated_seconds / 3600.0, 3),
                "benchmark_validation_auc": float(result["best_row"]["validation_auc"]),
                "benchmark_feature_count": int(result["feature_count"]),
            }
        )

    return {
        "sample_rows": int(len(modeling_sample_df)),
        "benchmark_sample_rows": int(len(benchmark_df)),
        "split_counts": {
            str(key): int(value)
            for key, value in modeling_sample_df["split_label"].value_counts(dropna=False).items()
        },
        "benchmark_split_counts": {
            str(key): int(value)
            for key, value in benchmark_df["split_label"].value_counts(dropna=False).items()
        },
        "per_model": model_rows,
        "estimated_total_grid_seconds": round(total_estimated_seconds, 3),
        "estimated_total_grid_hours": round(total_estimated_seconds / 3600.0, 3),
        "candidate_count_total": int(sum(row["grid_candidates"] for row in model_rows)),
        "threshold_hours": threshold_hours,
        "exceeds_threshold": bool(total_estimated_seconds > threshold_hours * 3600),
        "estimate_method": "capped benchmark sample scaled to full modeling sample with safety factor",
    }


def _sample_training_time_benchmark_frame(
    modeling_sample_df,
    max_train_rows,
    max_validation_rows,
    max_test_rows,
    negative_to_positive_ratio,
    random_state,
):
    sampled_parts = []
    for offset, (split_name, max_rows) in enumerate(
        [
            ("train", max_train_rows),
            ("validation", max_validation_rows),
            ("test", max_test_rows),
        ]
    ):
        split_df = modeling_sample_df[modeling_sample_df["split_label"] == split_name]
        positives = split_df[split_df["target_1m"].fillna(0).astype(int) == 1]
        negatives = split_df[split_df["target_1m"].fillna(0).astype(int) == 0]
        max_rows = int(max_rows)
        if len(positives) and len(negatives):
            target_positive_rows = min(
                len(positives),
                max(1, max_rows // (int(negative_to_positive_ratio) + 1)),
            )
            target_negative_rows = min(
                len(negatives),
                max_rows - target_positive_rows,
                target_positive_rows * int(negative_to_positive_ratio),
            )
        else:
            target_positive_rows = min(len(positives), max_rows)
            target_negative_rows = min(len(negatives), max_rows - target_positive_rows)
        if target_positive_rows < len(positives):
            positives = positives.sample(
                n=target_positive_rows,
                random_state=random_state + offset + 100,
            )
        if target_negative_rows < len(negatives):
            negatives = negatives.sample(
                n=target_negative_rows,
                random_state=random_state + offset,
            )
        split_sample = pd.concat([positives, negatives], ignore_index=True)
        if len(split_sample) > max_rows:
            split_sample = split_sample.sample(n=max_rows, random_state=random_state + offset)
        sampled_parts.append(split_sample)
    scoring = modeling_sample_df[modeling_sample_df["split_label"] == "scoring"]
    if not scoring.empty:
        sampled_parts.append(scoring.head(min(len(scoring), 10_000)))
    return pd.concat(sampled_parts, ignore_index=True).reset_index(drop=True)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "figures" / "temporal_model").mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "figures" / "loss_reserve").mkdir(parents=True, exist_ok=True)

    pd_config = build_pd_config()
    if pd_config.get("calendar_bigdata_backend") == "dask":
        if pd_config.get("calendar_bigdata_mode") == "sample_only":
            required_outputs = [
                Path(pd_config["calendar_modeling_sample_path"]),
                Path(pd_config["calendar_modeling_sample_summary_path"]),
                Path(pd_config["calendar_panel_summary_path"]),
                Path(pd_config["calendar_panel_counts_path"]),
            ]
            if pd_config.get("reuse_calendar_bigdata_outputs") and all(
                path.exists() for path in required_outputs
            ):
                print("Reusing existing Dask sample-only outputs...", flush=True)
                panel_summary = json.loads(
                    Path(pd_config["calendar_panel_summary_path"]).read_text(encoding="utf-8")
                )
                sample_summary = json.loads(
                    Path(pd_config["calendar_modeling_sample_summary_path"]).read_text(
                        encoding="utf-8"
                    )
                )
            else:
                print("Building calendar-time counts and sampled panel with Dask...", flush=True)
                stale_panel_path = Path(pd_config["calendar_panel_path"])
                if stale_panel_path.is_dir():
                    shutil.rmtree(stale_panel_path)
                elif stale_panel_path.exists():
                    stale_panel_path.unlink()
                build_result = build_calendar_sample_only_with_dask(
                    pd_config["processed_data_path"],
                    pd_config["calendar_modeling_sample_path"],
                    counts_output_path=pd_config["calendar_panel_counts_path"],
                    summary_output_path=pd_config["calendar_panel_summary_path"],
                    sample_summary_output_path=pd_config["calendar_modeling_sample_summary_path"],
                    data_cutoff=pd_config["data_cutoff"],
                    train_end=pd_config["temporal_split"]["train_end"],
                    valid_end=pd_config["temporal_split"]["valid_end"],
                    blocksize=pd_config.get("calendar_bigdata_blocksize", "64MB"),
                    **pd_config["calendar_bigdata_sample"],
                )
                panel_summary = build_result["panel_summary"]
                sample_summary = build_result["sample_summary"]
        else:
            print("Building calendar-time panel with Dask...", flush=True)
            panel_summary = build_calendar_panel_with_dask(
                pd_config["processed_data_path"],
                pd_config["calendar_panel_path"],
                counts_output_path=pd_config["calendar_panel_counts_path"],
                summary_output_path=pd_config["calendar_panel_summary_path"],
                data_cutoff=pd_config["data_cutoff"],
                train_end=pd_config["temporal_split"]["train_end"],
                valid_end=pd_config["temporal_split"]["valid_end"],
                blocksize=pd_config.get("calendar_bigdata_blocksize", "64MB"),
            )
            print(f"Calendar panel rows: {panel_summary['panel_rows']:,}", flush=True)
            print("Building deterministic modeling sample from Dask panel...", flush=True)
            from calendar_bigdata import build_modeling_sample_from_panel_dask

            sample_summary = build_modeling_sample_from_panel_dask(
                pd_config["calendar_panel_path"],
                pd_config["calendar_modeling_sample_path"],
                **pd_config["calendar_bigdata_sample"],
            )
            Path(pd_config["calendar_modeling_sample_summary_path"]).write_text(
                json.dumps(sample_summary, indent=2, default=str),
                encoding="utf-8",
            )
        print(
            f"Modeling sample rows: {sample_summary['rows']:,}; "
            f"splits={sample_summary['split_counts']}",
            flush=True,
        )
        raw_df = read_modeling_sample(pd_config["calendar_modeling_sample_path"])
        estimate_path = Path(pd_config["calendar_training_time_estimate_path"])
        if pd_config.get("reuse_calendar_bigdata_outputs") and estimate_path.exists():
            print("Reusing existing calendar training time estimate...", flush=True)
            time_estimate = json.loads(estimate_path.read_text(encoding="utf-8"))
        else:
            time_estimate = estimate_calendar_training_time(pd_config, raw_df)
            estimate_path.write_text(
                json.dumps(time_estimate, indent=2, default=str),
                encoding="utf-8",
            )
        print(
            f"Estimated full 62-candidate time: "
            f"{time_estimate['estimated_total_grid_hours']:.2f} hours",
            flush=True,
        )
        if time_estimate["exceeds_threshold"]:
            print(
                "Estimated training time exceeds 8 hours; stopping before full grid training.",
                flush=True,
            )
            print(json.dumps(time_estimate, indent=2, default=str), flush=True)
            if not pd_config.get("run_resource_bounded_after_estimate_exceeds_threshold", False):
                return
            bounded_sample_config = copy.deepcopy(
                pd_config.get("resource_bounded_training_sample", {})
            )
            print(
                "Running resource-bounded fixed-parameter benchmark for reports and paper...",
                flush=True,
            )
            raw_df = _sample_training_time_benchmark_frame(
                raw_df,
                max_train_rows=int(bounded_sample_config.get("max_train_rows", 10_000)),
                max_validation_rows=int(
                    bounded_sample_config.get("max_validation_rows", 5_000)
                ),
                max_test_rows=int(bounded_sample_config.get("max_test_rows", 25_000)),
                negative_to_positive_ratio=int(
                    bounded_sample_config.get("negative_to_positive_ratio", 5)
                ),
                random_state=int(bounded_sample_config.get("random_state", 42)),
            )
            bounded_sample_path = Path(pd_config["calendar_resource_bounded_sample_path"])
            bounded_summary_path = Path(
                pd_config["calendar_resource_bounded_sample_summary_path"]
            )
            bounded_sample_path.parent.mkdir(parents=True, exist_ok=True)
            raw_df.to_parquet(bounded_sample_path, index=False)
            bounded_summary = {
                "path": str(bounded_sample_path),
                "rows": int(len(raw_df)),
                "split_counts": {
                    str(key): int(value)
                    for key, value in raw_df["split_label"].value_counts(
                        dropna=False
                    ).items()
                },
                "event_counts": {
                    str(key): int(value)
                    for key, value in raw_df.groupby("split_label")["target_1m"]
                    .sum()
                    .items()
                },
                "source_modeling_sample_path": pd_config["calendar_modeling_sample_path"],
                "sampling_config": bounded_sample_config,
                "purpose": "resource_bounded_fixed_parameter_run_after_full_grid_estimate_exceeded_threshold",
            }
            bounded_summary_path.write_text(
                json.dumps(bounded_summary, indent=2, default=str),
                encoding="utf-8",
            )
            pd_config["calendar_modeling_sample_path"] = str(bounded_sample_path)
            pd_config["calendar_modeling_sample_summary_path"] = str(bounded_summary_path)
            experiment_config = copy.deepcopy(pd_config)
            experiment_config["calendar_panel_is_prebuilt"] = True
            experiment_config["use_grid_search"] = False
            experiment_config["grid_search_models"] = []
            experiment_config["resource_bounded_training"] = True
            experiment_config["resource_bounded_model_params_adjusted"] = True
            experiment_config["model_params"]["logistic_regression"]["max_iter"] = 300
            experiment_config["model_params"]["random_forest"].update(
                {"n_estimators": 40, "max_depth": 10}
            )
            experiment_config["model_params"]["xgboost"].update(
                {"n_estimators": 40, "max_depth": 4}
            )
            experiment_config["model_params"]["mlp_neural_network"].update(
                {
                    "hidden_layer_sizes": (32,),
                    "max_iter": 30,
                    "n_iter_no_change": 5,
                }
            )
            experiment_config["mlp_sampling"].update(
                {
                    "negative_to_positive_ratio": 5,
                    "max_train_rows": int(
                        bounded_sample_config.get("max_train_rows", 5_000)
                    ),
                }
            )
            experiment_config["calendar_training_time_estimate_result"] = time_estimate
            experiment_config["full_grid_training_skipped_reason"] = (
                "estimated_62_candidate_runtime_exceeded_threshold"
            )
        else:
            experiment_config = copy.deepcopy(pd_config)
            experiment_config["calendar_panel_is_prebuilt"] = True
            experiment_config["calendar_training_time_estimate_result"] = time_estimate
        experiment_config.pop("calendar_panel_path", None)
        experiment_config.pop("calendar_panel_counts_path", None)
    else:
        print("Loading raw loan data...", flush=True)
        raw_df = load_loss_raw_dataset(pd_config["processed_data_path"])
        if "sample_id" not in raw_df.columns:
            raw_df["sample_id"] = np.arange(len(raw_df))
        experiment_config = pd_config

    print(f"Running main PD hazard experiment on {len(raw_df):,} rows...", flush=True)
    result = run_experiment(experiment_config, raw_df)
    print(f"Main PD champion: {result['selected_model_key']}", flush=True)
    champion_predictions = result["champion_predictions"]
    export_columns = [
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
            "predicted_hazard_1m",
            "predicted_pd",
            "predicted_pd_12m",
            "model_name",
        ]
        if column in champion_predictions.columns
    ]
    champion_predictions[export_columns].to_csv(pd_config["output_path"], index=False)

    selected_model = result["selected_model_key"]
    stage2_config = {
        "modeling_type": "calendar_time_hazard",
        "split_date_column": "snapshot_month",
        "data_cutoff": pd_config["data_cutoff"],
        "processed_data_path": pd_config["processed_data_path"],
        "calendar_panel_is_prebuilt": bool(pd_config.get("calendar_bigdata_backend") == "dask"),
        "calendar_modeling_sample_path": pd_config.get("calendar_modeling_sample_path"),
        "calendar_bigdata_backend": pd_config.get("calendar_bigdata_backend"),
        "calendar_bigdata_sample": copy.deepcopy(pd_config.get("calendar_bigdata_sample", {})),
        "random_state": pd_config["random_state"],
        "sample_frac": 1.0,
        "temporal_split": copy.deepcopy(pd_config["temporal_split"]),
        "enabled_models": [selected_model],
        "selected_model_key": selected_model,
        "meta_columns": copy.deepcopy(pd_config["meta_columns"]),
        "model_params": {
            selected_model: copy.deepcopy(result["best_params_by_model"][selected_model])
        },
        "use_grid_search": False,
        "grid_search_models": [],
        "param_grid": {},
        "feature_profile": result["final_models"][selected_model].get("feature_profile"),
        "feature_count": len(result["final_models"][selected_model].get("feature_columns", [])),
        "pd_output": "predicted_pd_lifetime_calendar_time_hazard",
        "pd_12m_output": "predicted_pd_12m",
        "hazard_feature_mode": pd_config["hazard_feature_mode"],
        "mlp_sampling": copy.deepcopy(pd_config["mlp_sampling"]),
        "resource_bounded_training": bool(
            experiment_config.get("resource_bounded_training", False)
        ),
        "resource_bounded_model_params_adjusted": bool(
            experiment_config.get("resource_bounded_model_params_adjusted", False)
        ),
        "full_grid_training_skipped_reason": experiment_config.get(
            "full_grid_training_skipped_reason"
        ),
        "calendar_training_time_estimate_path": pd_config.get(
            "calendar_training_time_estimate_path"
        ),
        "estimated_full_grid_hours": (
            experiment_config.get("calendar_training_time_estimate_result", {}) or {}
        ).get("estimated_total_grid_hours"),
        "max_stage2_scoring_rows_per_scope": 5_000
        if experiment_config.get("resource_bounded_training", False)
        else None,
    }
    Path(pd_config["stage2_champion_config_path"]).write_text(
        json.dumps(stage2_config, indent=2),
        encoding="utf-8",
    )

    print("Exporting temporal report and figures...", flush=True)
    temporal_visuals = export_temporal_visuals(
        result,
        output_dir=REPORT_DIR / "figures" / "temporal_model",
        feature_search_result=None,
        show=False,
    )
    write_temporal_report(
        result,
        experiment_config,
        pd_config["report_output_path"],
        feature_search_result=None,
        visual_paths=temporal_visuals,
    )

    print("Running loss reserve workflow with the new PD champion...", flush=True)
    loss_config = build_loss_config(stage2_config)
    loss_result = run_loss_reserve_workflow(loss_config)
    loss_visuals = export_loss_reserve_visuals(
        loss_result,
        output_dir=REPORT_DIR / "figures" / "loss_reserve",
        show=False,
    )
    print(json.dumps({
        "selected_model": selected_model,
        "champion_validation": result["champion_validation"],
        "champion_test": result["champion_test"],
        "pd_predictions_path": pd_config["output_path"],
        "stage2_config_path": pd_config["stage2_champion_config_path"],
        "temporal_report_path": pd_config["report_output_path"],
        "loss_report_path": str(loss_result["report_path"]),
        "loss_visual_count": len(loss_visuals),
    }, default=str, indent=2))


if __name__ == "__main__":
    main()
