import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from calendar_bigdata import read_modeling_sample
from experiment import fit_locked_model, score_trained_model
from hazard import (
    DEFAULT_HAZARD_PARAM_GRID,
    DEFAULT_HGB_HAZARD_PARAM_GRID,
    fit_time_consistent_hazard,
    score_hazard_probabilities,
)
from loss_preprocess import (
    ACTIVE_LOAN_STATUSES,
    CHARGED_OFF_STATUS,
    LOSS_CLEANING_METHODS,
    RESOLVED_LOAN_STATUSES,
    SPLIT_LABEL_COLUMN,
    STATUS_COLUMN,
    apply_expected_lgd_lookup,
    build_charged_off_loss_proxy,
    build_managerial_segmentation_summary,
    build_portfolio_expected_loss_summary,
    compute_expected_loss,
    fit_expected_lgd_lookup,
    load_loss_raw_dataset,
    prepare_loss_workflow_dataset,
)
from loss_reporting import write_loss_reserve_report
from preprocess import ID_COLUMN, ISSUE_DATE_COLUMN, build_feature_dataset_from_raw, load_processed_data


def run_loss_reserve_workflow(config):
    raw_data_path = Path(config["raw_data_path"])
    pd_prediction_path = Path(config["pd_prediction_path"])
    loss_workflow_path = Path(config["loss_workflow_path"])
    charged_off_proxy_path = Path(config["charged_off_proxy_path"])
    cecl_snapshot_path = Path(config["cecl_snapshot_path"])
    test_loss_metrics_path = Path(config["test_loss_metrics_path"])
    segment_summary_path = Path(config["segment_summary_path"])
    portfolio_summary_path = Path(config["portfolio_summary_path"])
    hazard_comparison_path = Path(
        config.get(
            "hazard_comparison_path",
            portfolio_summary_path.parent / "hazard_model_comparison.csv",
        )
    )
    hazard_search_path = Path(
        config.get(
            "hazard_search_path",
            portfolio_summary_path.parent / "hazard_grid_search_results.csv",
        )
    )
    report_output_path = Path(config["report_output_path"])

    raw_loss_df = load_loss_raw_dataset(
        raw_data_path,
        **config.get("raw_read_csv_kwargs", {}),
    )
    if ID_COLUMN not in raw_loss_df.columns:
        raw_loss_df[ID_COLUMN] = np.arange(len(raw_loss_df))

    workflow_df = prepare_loss_workflow_dataset(
        raw_loss_df,
        train_end=config["temporal_split"]["train_end"],
        valid_end=config["temporal_split"]["valid_end"],
    )
    _write_csv(workflow_df, loss_workflow_path)

    charged_off_proxy = build_charged_off_loss_proxy(workflow_df)
    _write_csv(charged_off_proxy, charged_off_proxy_path)

    lgd_lookup = fit_expected_lgd_lookup(
        charged_off_proxy,
        fit_splits=tuple(config.get("lgd_fit_splits", ["train", "validation"])),
        shrinkage_k=int(config.get("lgd_shrinkage_k", 50)),
    )

    hazard_result = fit_time_consistent_hazard(
        workflow_df,
        logistic_param_grid=config.get("hazard_param_grid", DEFAULT_HAZARD_PARAM_GRID),
        ml_param_grid=config.get("hazard_ml_param_grid", DEFAULT_HGB_HAZARD_PARAM_GRID),
    )
    _write_csv(hazard_result["search_table"], hazard_search_path)
    _write_csv(hazard_result["model_comparison_table"], hazard_comparison_path)

    resolved_holdout = workflow_df[
        (workflow_df[SPLIT_LABEL_COLUMN] == "test")
        & (workflow_df[STATUS_COLUMN].isin(RESOLVED_LOAN_STATUSES))
    ].copy()
    resolved_holdout = score_hazard_probabilities(
        hazard_result["final_bundle"],
        resolved_holdout,
        start_month_column=None,
    )
    resolved_holdout = apply_expected_lgd_lookup(resolved_holdout, lgd_lookup)
    resolved_holdout["ead_reference"] = resolved_holdout["funded_amnt"].fillna(0).clip(lower=0)
    resolved_holdout = compute_expected_loss(
        resolved_holdout,
        pd_column="pd_12m",
        lgd_column="expected_lgd",
        ead_column="ead_reference",
        output_column="expected_loss_12m",
    ).rename(columns={"pd_12m": "pd_12m_hazard"})
    resolved_holdout = compute_expected_loss(
        resolved_holdout,
        pd_column="lifetime_pd",
        lgd_column="expected_lgd",
        ead_column="ead_reference",
        output_column="expected_loss_lifetime",
    ).rename(columns={"lifetime_pd": "lifetime_pd_hazard"})

    charged_holdout_metrics = charged_off_proxy[
        charged_off_proxy[SPLIT_LABEL_COLUMN] == "test"
    ][
        [
            "sample_id",
            "ead_proxy",
            "lgd_proxy",
            "recovery_rate_proxy",
            "realized_loss_amount",
        ]
    ].copy()
    resolved_holdout = resolved_holdout.merge(
        charged_holdout_metrics,
        on="sample_id",
        how="left",
    )
    for column in ["ead_proxy", "lgd_proxy", "recovery_rate_proxy", "realized_loss_amount"]:
        resolved_holdout[column] = resolved_holdout[column].fillna(0)

    active_snapshot = workflow_df[workflow_df[STATUS_COLUMN].isin(ACTIVE_LOAN_STATUSES)].copy()
    active_snapshot = score_hazard_probabilities(
        hazard_result["final_bundle"],
        active_snapshot,
        start_month_column="months_on_book",
    )
    active_snapshot = apply_expected_lgd_lookup(active_snapshot, lgd_lookup)
    active_snapshot = compute_expected_loss(
        active_snapshot.rename(columns={"pd_12m": "pd_12m_hazard"}),
        pd_column="pd_12m_hazard",
        lgd_column="expected_lgd",
        ead_column="ead_current",
        output_column="expected_loss_12m",
    )
    active_snapshot = compute_expected_loss(
        active_snapshot.rename(columns={"lifetime_pd": "lifetime_pd_hazard"}),
        pd_column="lifetime_pd_hazard",
        lgd_column="expected_lgd",
        ead_column="ead_current",
        output_column="lifetime_expected_loss",
    )

    observable_12m_holdout = workflow_df[
        (workflow_df[SPLIT_LABEL_COLUMN] == "test")
        & (workflow_df[ISSUE_DATE_COLUMN] < pd.Timestamp("2018-01-01"))
    ].copy()
    observable_12m_holdout = apply_expected_lgd_lookup(observable_12m_holdout, lgd_lookup)
    observable_12m_holdout["ead_reference"] = (
        observable_12m_holdout["funded_amnt"].fillna(0).clip(lower=0)
    )

    stage2_benchmark_summary = {}
    stage2_scoring_config = config.get("stage2_scoring", {})
    if stage2_scoring_config and stage2_scoring_config.get("precomputed_dual_pd_prediction_path"):
        stage2_predictions = pd.read_csv(stage2_scoring_config["precomputed_dual_pd_prediction_path"])
        resolved_stage2_predictions = _stage2_scope_predictions(
            stage2_predictions,
            "resolved_test",
        )
        active_stage2_predictions = _stage2_scope_predictions(
            stage2_predictions,
            "active_snapshot",
        )
        observable_12m_predictions = _stage2_scope_predictions(
            stage2_predictions,
            "observable_12m_test",
        )
        resolved_holdout = _merge_stage2_predictions(resolved_holdout, resolved_stage2_predictions)
        active_snapshot = _merge_stage2_predictions(active_snapshot, active_stage2_predictions)
        observable_12m_holdout = _merge_stage2_predictions(
            observable_12m_holdout,
            observable_12m_predictions,
        )
        combined_stage2 = pd.concat(
            [resolved_stage2_predictions, active_stage2_predictions, observable_12m_predictions],
            ignore_index=True,
        )
        direct_12m_model = stage2_scoring_config.get("direct_12m_model", {})
        hazard_comparison_model = stage2_scoring_config.get(
            "hazard_12m_comparison_model",
            stage2_scoring_config.get("hazard_12m_model", {}),
        )
        prediction_source = "precomputed_dual_pd_static_lifetime_and_direct_active_snapshot_12m"
        if not direct_12m_model:
            prediction_source = "precomputed_dual_pd_static_lifetime_and_12m_hazard"
        stage2_benchmark_summary = {
            "model_name": stage2_scoring_config.get("model_name", "Dual PD"),
            "rows": int(combined_stage2["sample_id"].nunique()),
            "mean_predicted_pd": float(combined_stage2["stage2_champion_pd"].mean()),
            "mean_predicted_pd_12m": float(combined_stage2["stage2_champion_pd_12m"].mean()),
            "mean_predicted_pd_lifetime": float(combined_stage2["stage2_champion_pd_lifetime"].mean()),
            "prediction_source": prediction_source,
            "used_for_12m_el": True,
            "used_for_lifetime_el": True,
            "full_stage2_scoring": True,
            "resolved_stage2_scored_rows": int(resolved_stage2_predictions["sample_id"].nunique()),
            "active_stage2_scored_rows": int(active_stage2_predictions["sample_id"].nunique()),
            "observable_12m_stage2_scored_rows": int(observable_12m_predictions["sample_id"].nunique()),
            "unscored_rows_fallback_to_reserve_hazard": False,
            "lifetime_model": stage2_scoring_config.get("lifetime_model", {}),
            "direct_12m_model": direct_12m_model,
            "hazard_12m_comparison_model": hazard_comparison_model,
            "hazard_12m_model": hazard_comparison_model,
        }
    elif stage2_scoring_config:
        if stage2_scoring_config.get("calendar_panel_is_prebuilt"):
            stage2_training_df = read_modeling_sample(
                stage2_scoring_config["calendar_modeling_sample_path"]
            )
        elif stage2_scoring_config.get("modeling_type") in {"hazard", "calendar_time_hazard"}:
            stage2_training_df = load_loss_raw_dataset(
                stage2_scoring_config["processed_data_path"]
            )
            if ID_COLUMN not in stage2_training_df.columns:
                stage2_training_df[ID_COLUMN] = np.arange(len(stage2_training_df))
        else:
            stage2_training_df = load_processed_data(
                stage2_scoring_config["processed_data_path"]
            )
        stage2_bundle = fit_locked_model(
            stage2_scoring_config,
            stage2_training_df,
            model_name=stage2_scoring_config.get("selected_model_key"),
        )
        stage2_score_cap = stage2_scoring_config.get("max_stage2_scoring_rows_per_scope")
        if bool(stage2_scoring_config.get("full_stage2_scoring", False)):
            stage2_score_cap = None
        resolved_stage2_ids = _cap_stage2_sample_ids(
            resolved_holdout["sample_id"],
            max_rows=stage2_score_cap,
            random_state=int(stage2_scoring_config.get("random_state", 42)),
        )
        active_stage2_ids = _cap_stage2_sample_ids(
            active_snapshot["sample_id"],
            max_rows=stage2_score_cap,
            random_state=int(stage2_scoring_config.get("random_state", 42)) + 1,
        )
        scoring_start = time.perf_counter()
        parallel_jobs = int(stage2_scoring_config.get("stage2_scoring_parallel_jobs", 1) or 1)
        if parallel_jobs >= 2 and stage2_score_cap is None:
            with ThreadPoolExecutor(max_workers=2) as executor:
                resolved_future = executor.submit(
                    _score_stage2_subset,
                    stage2_bundle,
                    raw_loss_df,
                    resolved_stage2_ids,
                )
                active_future = executor.submit(
                    _score_stage2_subset,
                    stage2_bundle,
                    raw_loss_df,
                    active_stage2_ids,
                )
                resolved_stage2_predictions = resolved_future.result()
                active_stage2_predictions = active_future.result()
        else:
            resolved_stage2_predictions = _score_stage2_subset(
                stage2_bundle,
                raw_loss_df,
                resolved_stage2_ids,
            )
            active_stage2_predictions = _score_stage2_subset(
                stage2_bundle,
                raw_loss_df,
                active_stage2_ids,
            )
        scoring_seconds = time.perf_counter() - scoring_start
        resolved_holdout = resolved_holdout.merge(
            resolved_stage2_predictions.rename(
                columns={
                    "predicted_pd": "stage2_champion_pd",
                    "predicted_pd_12m": "stage2_champion_pd_12m",
                    "predicted_pd_lifetime": "stage2_champion_pd_lifetime",
                    "model_name": "stage2_model_name",
                }
            ),
            on="sample_id",
            how="left",
        )
        active_snapshot = active_snapshot.merge(
            active_stage2_predictions.rename(
                columns={
                    "predicted_pd": "stage2_champion_pd",
                    "predicted_pd_12m": "stage2_champion_pd_12m",
                    "predicted_pd_lifetime": "stage2_champion_pd_lifetime",
                    "model_name": "stage2_model_name",
                }
            ),
            on="sample_id",
            how="left",
        )
        combined_stage2 = pd.concat(
            [resolved_stage2_predictions, active_stage2_predictions],
            ignore_index=True,
        )
        stage2_benchmark_summary = {
            "model_name": stage2_bundle["model_name"],
            "rows": int(combined_stage2["sample_id"].nunique()),
            "mean_predicted_pd": float(combined_stage2["predicted_pd"].mean()),
            "mean_predicted_pd_12m": float(combined_stage2["predicted_pd_12m"].mean())
            if "predicted_pd_12m" in combined_stage2.columns and not combined_stage2.empty
            else np.nan,
            "mean_predicted_pd_lifetime": float(combined_stage2["predicted_pd_lifetime"].mean())
            if "predicted_pd_lifetime" in combined_stage2.columns and not combined_stage2.empty
            else np.nan,
            "prediction_source": "locked_calendar_time_hazard_pd_champion"
            if stage2_bundle.get("modeling_type") == "calendar_time_hazard"
            else "locked_hazard_pd_champion"
            if stage2_bundle.get("modeling_type") == "hazard"
            else "locked_static_pd_champion",
            "used_for_12m_el": True,
            "max_stage2_scoring_rows_per_scope": stage2_score_cap,
            "full_stage2_scoring": bool(stage2_scoring_config.get("full_stage2_scoring", False)),
            "stage2_scoring_seconds": float(scoring_seconds),
            "stage2_scoring_parallel_jobs": parallel_jobs if stage2_score_cap is None else 1,
            "resolved_stage2_scored_rows": int(
                resolved_stage2_predictions["sample_id"].nunique()
            )
            if "sample_id" in resolved_stage2_predictions.columns
            else 0,
            "active_stage2_scored_rows": int(
                active_stage2_predictions["sample_id"].nunique()
            )
            if "sample_id" in active_stage2_predictions.columns
            else 0,
            "unscored_rows_fallback_to_reserve_hazard": bool(stage2_score_cap),
            "hazard_calibration": stage2_bundle.get("hazard_calibration", {}),
        }
        estimate_path = stage2_scoring_config.get("stage2_time_estimate_path")
        if estimate_path:
            _write_json(
                {
                    "actual_stage2_scoring_seconds": float(scoring_seconds),
                    "resolved_stage2_scored_rows": stage2_benchmark_summary[
                        "resolved_stage2_scored_rows"
                    ],
                    "active_stage2_scored_rows": stage2_benchmark_summary[
                        "active_stage2_scored_rows"
                    ],
                    "stage2_scoring_parallel_jobs": stage2_benchmark_summary[
                        "stage2_scoring_parallel_jobs"
                    ],
                    "hazard_calibration": stage2_benchmark_summary["hazard_calibration"],
                },
                estimate_path,
                merge_existing=True,
            )
    elif pd_prediction_path.exists():
        stage2_predictions = pd.read_csv(pd_prediction_path)
        stage2_predictions = stage2_predictions.rename(
            columns={"predicted_pd": "stage2_champion_pd", "model_name": "stage2_model_name"}
        )
        resolved_holdout = resolved_holdout.merge(
            stage2_predictions[["sample_id", "stage2_champion_pd", "stage2_model_name"]],
            on="sample_id",
            how="left",
        )
        stage2_benchmark_summary = {
            "model_name": stage2_predictions["stage2_model_name"].mode().iat[0]
            if "stage2_model_name" in stage2_predictions.columns and not stage2_predictions.empty
            else "NA",
            "rows": int(stage2_predictions["sample_id"].nunique()),
            "mean_predicted_pd": float(stage2_predictions["stage2_champion_pd"].mean()),
            "prediction_source": "merged_test_predictions_only",
            "used_for_12m_el": False,
        }

    for frame, ead_column in [
        (resolved_holdout, "ead_reference"),
        (active_snapshot, "ead_current"),
    ]:
        if "stage2_champion_pd" not in frame.columns:
            frame["stage2_champion_pd"] = np.nan
        if "stage2_champion_pd_12m" not in frame.columns:
            frame["stage2_champion_pd_12m"] = np.nan
        if "stage2_champion_pd_lifetime" not in frame.columns:
            frame["stage2_champion_pd_lifetime"] = np.nan
        frame["pd_12m_fixed_horizon"] = frame["stage2_champion_pd_12m"].fillna(
            frame["stage2_champion_pd"]
        ).fillna(
            frame["pd_12m_hazard"]
        )
        frame["lifetime_pd_hazard"] = frame["stage2_champion_pd_lifetime"].fillna(
            frame["lifetime_pd_hazard"]
        )

    if "stage2_champion_pd" not in observable_12m_holdout.columns:
        observable_12m_holdout["stage2_champion_pd"] = np.nan
    if "stage2_champion_pd_12m" not in observable_12m_holdout.columns:
        observable_12m_holdout["stage2_champion_pd_12m"] = np.nan
    if "stage2_champion_pd_lifetime" not in observable_12m_holdout.columns:
        observable_12m_holdout["stage2_champion_pd_lifetime"] = np.nan
    observable_12m_holdout["pd_12m_fixed_horizon"] = (
        observable_12m_holdout["stage2_champion_pd_12m"]
        .fillna(observable_12m_holdout["stage2_champion_pd"])
        .fillna(0)
    )
    observable_12m_holdout["lifetime_pd_hazard"] = (
        observable_12m_holdout["stage2_champion_pd_lifetime"]
        .fillna(observable_12m_holdout.get("lifetime_pd_hazard", np.nan))
    )

    observable_12m_holdout = compute_expected_loss(
        observable_12m_holdout,
        pd_column="pd_12m_fixed_horizon",
        lgd_column="expected_lgd",
        ead_column="ead_reference",
        output_column="expected_loss_12m",
    )
    resolved_holdout = compute_expected_loss(
        resolved_holdout,
        pd_column="pd_12m_fixed_horizon",
        lgd_column="expected_lgd",
        ead_column="ead_reference",
        output_column="expected_loss_12m",
    )
    active_snapshot = compute_expected_loss(
        active_snapshot,
        pd_column="pd_12m_fixed_horizon",
        lgd_column="expected_lgd",
        ead_column="ead_current",
        output_column="expected_loss_12m",
    )
    resolved_holdout = compute_expected_loss(
        resolved_holdout,
        pd_column="lifetime_pd_hazard",
        lgd_column="expected_lgd",
        ead_column="ead_reference",
        output_column="expected_loss_lifetime",
    )
    active_snapshot = compute_expected_loss(
        active_snapshot,
        pd_column="lifetime_pd_hazard",
        lgd_column="expected_lgd",
        ead_column="ead_current",
        output_column="lifetime_expected_loss",
    )

    _write_csv(resolved_holdout, test_loss_metrics_path)
    _write_csv(active_snapshot, cecl_snapshot_path)
    observable_path = config.get("observable_12m_loss_metrics_path") or stage2_scoring_config.get(
        "observable_12m_loss_metrics_path"
    )
    if observable_path:
        _write_csv(observable_12m_holdout, observable_path)

    portfolio_summary = pd.concat(
        [
            _build_scope_portfolio_summary(
                observable_12m_holdout,
                analysis_scope="observable_12m_test",
                pd_measure="12m",
                pd_column="pd_12m_fixed_horizon",
                lgd_column="expected_lgd",
                ead_column="ead_reference",
                el_column="expected_loss_12m",
                include_actual_loss=True,
            ),
            _build_scope_portfolio_summary(
                resolved_holdout,
                analysis_scope="resolved_test",
                pd_measure="12m",
                pd_column="pd_12m_fixed_horizon",
                lgd_column="expected_lgd",
                ead_column="ead_reference",
                el_column="expected_loss_12m",
                include_actual_loss=True,
            ),
            _build_scope_portfolio_summary(
                resolved_holdout,
                analysis_scope="resolved_test",
                pd_measure="lifetime",
                pd_column="lifetime_pd_hazard",
                lgd_column="expected_lgd",
                ead_column="ead_reference",
                el_column="expected_loss_lifetime",
                include_actual_loss=True,
            ),
            _build_scope_portfolio_summary(
                active_snapshot,
                analysis_scope="active_snapshot",
                pd_measure="12m",
                pd_column="pd_12m_fixed_horizon",
                lgd_column="expected_lgd",
                ead_column="ead_current",
                el_column="expected_loss_12m",
            ),
            _build_scope_portfolio_summary(
                active_snapshot,
                analysis_scope="active_snapshot",
                pd_measure="lifetime",
                pd_column="lifetime_pd_hazard",
                lgd_column="expected_lgd",
                ead_column="ead_current",
                el_column="lifetime_expected_loss",
            ),
        ],
        ignore_index=True,
    )
    _write_csv(portfolio_summary, portfolio_summary_path)

    segment_summary = pd.concat(
        [
            build_managerial_segmentation_summary(
                observable_12m_holdout,
                analysis_scope="observable_12m_test",
                pd_measure="12m",
                pd_column="pd_12m_fixed_horizon",
                lgd_column="expected_lgd",
                ead_column="ead_reference",
                el_column="expected_loss_12m",
            ),
            build_managerial_segmentation_summary(
                resolved_holdout,
                analysis_scope="resolved_test",
                pd_measure="12m",
                pd_column="pd_12m_fixed_horizon",
                lgd_column="expected_lgd",
                ead_column="ead_reference",
                el_column="expected_loss_12m",
            ),
            build_managerial_segmentation_summary(
                resolved_holdout,
                analysis_scope="resolved_test",
                pd_measure="lifetime",
                pd_column="lifetime_pd_hazard",
                lgd_column="expected_lgd",
                ead_column="ead_reference",
                el_column="expected_loss_lifetime",
            ),
            build_managerial_segmentation_summary(
                active_snapshot,
                analysis_scope="active_snapshot",
                pd_measure="12m",
                pd_column="pd_12m_fixed_horizon",
                lgd_column="expected_lgd",
                ead_column="ead_current",
                el_column="expected_loss_12m",
            ),
            build_managerial_segmentation_summary(
                active_snapshot,
                analysis_scope="active_snapshot",
                pd_measure="lifetime",
                pd_column="lifetime_pd_hazard",
                lgd_column="expected_lgd",
                ead_column="ead_current",
                el_column="lifetime_expected_loss",
            ),
        ],
        ignore_index=True,
    )
    _write_csv(segment_summary, segment_summary_path)

    lgd_holdout = resolved_holdout[resolved_holdout[STATUS_COLUMN] == CHARGED_OFF_STATUS].copy()
    lgd_lookup_summary = {
        "fit_splits": list(config.get("lgd_fit_splits", ["train", "validation"])),
        "shrinkage_k": int(config.get("lgd_shrinkage_k", 50)),
        "portfolio_mean": float(lgd_lookup["portfolio_mean"]),
        "training_rows": int(lgd_lookup["training_rows"]),
        "test_lgd_mae": float(np.mean(np.abs(lgd_holdout["expected_lgd"] - lgd_holdout["lgd_proxy"])))
        if not lgd_holdout.empty
        else np.nan,
        "test_lgd_rmse": float(
            np.sqrt(np.mean((lgd_holdout["expected_lgd"] - lgd_holdout["lgd_proxy"]) ** 2))
        )
        if not lgd_holdout.empty
        else np.nan,
    }

    lgd_lookup_source_table = pd.concat(
        [
            _source_usage_table(resolved_holdout, "resolved_test"),
            _source_usage_table(active_snapshot, "active_snapshot"),
        ],
        ignore_index=True,
    )

    workflow_result = {
        "loss_dataset_summary": _build_loss_dataset_summary(workflow_df),
        "charged_off_proxy_summary": {
            "rows": int(len(charged_off_proxy)),
            "portfolio_lgd_mean": float(charged_off_proxy["lgd_proxy"].mean()),
            "avg_ead_proxy": float(charged_off_proxy["ead_proxy"].mean()),
        },
        "lgd_lookup_summary": lgd_lookup_summary,
        "lgd_lookup_source_table": lgd_lookup_source_table,
        "hazard_panel_summary": hazard_result["panel_summary"],
        "hazard_search_table": hazard_result["search_table"],
        "hazard_model_comparison": hazard_result["model_comparison_table"],
        "selected_hazard_model": {
            "model_key": hazard_result["selected_model_key"],
            "model_name": hazard_result["selected_model_name"],
            "params": hazard_result["selected_params"],
            "search_path": str(hazard_search_path),
            "comparison_path": str(hazard_comparison_path),
        },
        "hazard_validation_overall": _build_hazard_metric_table(
            hazard_result["validation_metrics"],
            hazard_result["validation_metrics_12m"],
            split_name="validation",
            model_name=hazard_result["selected_model_name"],
        ),
        "hazard_test_overall": _build_hazard_metric_table(
            hazard_result["test_metrics"],
            hazard_result["test_metrics_12m"],
            split_name="test",
            model_name=hazard_result["selected_model_name"],
        ),
        "hazard_validation_vintage": hazard_result["validation_vintage_table"],
        "hazard_test_vintage": hazard_result["test_vintage_table"],
        "portfolio_summary": portfolio_summary,
        "segment_summary": segment_summary,
        "observable_12m_holdout_sample": observable_12m_holdout[
            [
                "sample_id",
                "issue_date",
                STATUS_COLUMN,
                "stage2_model_name",
                "stage2_champion_pd_12m",
                "pd_12m_fixed_horizon",
                "expected_lgd",
                "ead_reference",
                "expected_loss_12m",
                "actual_default_12m",
                "actual_net_loss",
            ]
        ]
        .head(10)
        .copy(),
        "resolved_holdout_sample": resolved_holdout.rename(
            columns={"hazard_model_name": "auxiliary_hazard_model_name"}
        )[
            [
                "sample_id",
                "issue_date",
                STATUS_COLUMN,
                "stage2_model_name",
                "auxiliary_hazard_model_name",
                "stage2_champion_pd",
                "pd_12m_fixed_horizon",
                "pd_12m_hazard",
                "lifetime_pd_hazard",
                "expected_lgd",
                "ead_reference",
                "expected_loss_12m",
                "expected_loss_lifetime",
                "actual_net_loss",
            ]
        ]
        .head(10)
        .copy(),
        "active_snapshot_sample": active_snapshot.rename(
            columns={"hazard_model_name": "auxiliary_hazard_model_name"}
        )[
            [
                "sample_id",
                "issue_date",
                STATUS_COLUMN,
                "remaining_term",
                "stage2_model_name",
                "auxiliary_hazard_model_name",
                "stage2_champion_pd",
                "pd_12m_fixed_horizon",
                "pd_12m_hazard",
                "lifetime_pd_hazard",
                "expected_lgd",
                "ead_current",
                "expected_loss_12m",
                "lifetime_expected_loss",
            ]
        ]
        .head(10)
        .copy(),
        "stage2_benchmark_summary": stage2_benchmark_summary,
    }

    report_path = write_loss_reserve_report(
        workflow_result,
        report_config=config,
        output_path=report_output_path,
    )

    return {
        "workflow_df": workflow_df,
        "charged_off_proxy": charged_off_proxy,
        "resolved_holdout": resolved_holdout,
        "active_snapshot": active_snapshot,
        "observable_12m_holdout": observable_12m_holdout,
        "portfolio_summary": portfolio_summary,
        "segment_summary": segment_summary,
        "workflow_result": workflow_result,
        "hazard_result": hazard_result,
        "lgd_lookup": lgd_lookup,
        "report_path": report_path,
    }


def _write_csv(df, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def _stage2_scope_predictions(stage2_predictions, scope):
    if "prediction_scope" in stage2_predictions.columns:
        scoped = stage2_predictions[stage2_predictions["prediction_scope"].eq(scope)].copy()
    else:
        scoped = stage2_predictions.copy()
    rename_map = {
        "predicted_pd": "stage2_champion_pd",
        "predicted_pd_12m": "stage2_champion_pd_12m",
        "predicted_pd_lifetime": "stage2_champion_pd_lifetime",
        "model_name": "stage2_model_name",
    }
    scoped = scoped.rename(columns=rename_map)
    keep_columns = [
        column
        for column in [
            "sample_id",
            "stage2_champion_pd",
            "stage2_champion_pd_12m",
            "stage2_champion_pd_lifetime",
            "stage2_model_name",
            "predicted_pd_lifetime_static_raw",
            "predicted_pd_lifetime_static",
            "predicted_pd_12m_raw",
            "predicted_pd_12m_model_calibrated",
            "pd_12m_conservative_floor",
            "predicted_hazard_1m",
            "predicted_pd_12m_direct_raw",
            "predicted_pd_12m_direct_calibrated",
            "predicted_pd_12m_direct",
            "pd_12m_direct_conservative_floor",
            "predicted_pd_12m_hazard_raw",
            "predicted_pd_12m_hazard",
            "predicted_hazard_1m_hazard",
            "lifetime_model_name",
            "hazard_12m_model_name",
            "direct_12m_model_name",
            "lifetime_model_source",
            "pd_12m_model_source",
            "pd_12m_champion_source",
            "prediction_scope",
        ]
        if column in scoped.columns
    ]
    return scoped[keep_columns].copy()


def _merge_stage2_predictions(frame, predictions):
    merge_columns = [
        column
        for column in predictions.columns
        if column == "sample_id" or column not in frame.columns
    ]
    return frame.merge(predictions[merge_columns], on="sample_id", how="left")


def _write_json(payload, output_path, merge_existing=False):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if merge_existing and output_path.exists():
        try:
            existing = json.loads(output_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
        existing.update(payload)
        payload = existing
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _cap_stage2_sample_ids(sample_ids, max_rows=None, random_state=42):
    unique_ids = pd.Series(sample_ids).dropna().drop_duplicates()
    if max_rows is None:
        return unique_ids
    max_rows = int(max_rows)
    if max_rows <= 0 or len(unique_ids) <= max_rows:
        return unique_ids
    return unique_ids.sample(n=max_rows, random_state=random_state)


def _build_loss_dataset_summary(df):
    return (
        df.groupby([SPLIT_LABEL_COLUMN, STATUS_COLUMN], as_index=False)
        .agg(
            loan_count=("sample_id", "size"),
            funded_amount=("funded_amnt", "sum"),
            avg_months_on_book=("months_on_book", "mean"),
        )
        .reset_index(drop=True)
    )


def _build_scope_portfolio_summary(
    df,
    analysis_scope,
    pd_measure,
    pd_column,
    lgd_column,
    ead_column,
    el_column,
    include_actual_loss=False,
):
    summary = build_portfolio_expected_loss_summary(
        df,
        analysis_scope=analysis_scope,
        pd_measure=pd_measure,
        pd_column=pd_column,
        lgd_column=lgd_column,
        ead_column=ead_column,
        el_column=el_column,
    )
    actual_df = df
    if (
        include_actual_loss
        and pd_measure == "12m"
        and "actual_default_12m" in df.columns
    ):
        actual_df = df[df["actual_default_12m"].fillna(0).astype(int) == 1]
    summary["actual_loss_amount"] = (
        float(actual_df["actual_net_loss"].sum())
        if include_actual_loss and "actual_net_loss" in df.columns
        else np.nan
    )
    summary["actual_loss_rate"] = (
        float(actual_df["actual_net_loss"].sum() / df["funded_amnt"].sum())
        if include_actual_loss and "actual_net_loss" in df.columns and df["funded_amnt"].sum() > 0
        else np.nan
    )
    return summary


def _source_usage_table(df, analysis_scope):
    if "lgd_lookup_source" not in df.columns:
        return pd.DataFrame()
    usage = (
        df.groupby("lgd_lookup_source", as_index=False)
        .agg(loan_count=("sample_id", "size"))
        .reset_index(drop=True)
    )
    usage["analysis_scope"] = analysis_scope
    return usage


def _build_hazard_metric_table(lifetime_metrics, metrics_12m, split_name, model_name=None):
    return pd.DataFrame(
        [
            {
                "split_name": split_name,
                "hazard_model_name": model_name,
                "pd_measure": "lifetime",
                "auc": float(lifetime_metrics["AUC"]),
                "ks": float(lifetime_metrics["KS"]),
                "brier": float(lifetime_metrics["Brier"]),
            },
            {
                "split_name": split_name,
                "hazard_model_name": model_name,
                "pd_measure": "12m",
                "auc": float(metrics_12m["AUC"]),
                "ks": float(metrics_12m["KS"]),
                "brier": float(metrics_12m["Brier"]),
            },
        ]
    )


def _score_stage2_subset(stage2_bundle, raw_loss_df, sample_ids):
    if len(sample_ids) == 0:
        return pd.DataFrame(columns=["sample_id", "predicted_pd", "model_name"])

    raw_subset = raw_loss_df[raw_loss_df[ID_COLUMN].isin(sample_ids)].copy()
    raw_subset[ID_COLUMN] = raw_subset[ID_COLUMN].astype(int)
    if stage2_bundle.get("modeling_type") == "calendar_time_hazard":
        issue_month = pd.to_datetime(raw_subset["issue_d"], format="%b-%Y", errors="coerce")
        raw_subset["snapshot_month"] = issue_month
        raw_subset["months_on_book"] = 0
        active_mask = raw_subset[STATUS_COLUMN].isin(ACTIVE_LOAN_STATUSES)
        data_cutoff = pd.Timestamp(
            stage2_bundle.get("data_cutoff", "2018-12-01")
        ).to_period("M").to_timestamp()
        raw_subset.loc[active_mask, "snapshot_month"] = data_cutoff
        raw_subset.loc[active_mask, "months_on_book"] = _month_diff_to_timestamp(
            issue_month.loc[active_mask],
            data_cutoff,
        )
        predictions = score_trained_model(stage2_bundle, raw_subset)
        if "predicted_pd_12m" in predictions.columns:
            predictions["predicted_pd_lifetime"] = predictions["predicted_pd"]
            predictions["predicted_pd"] = predictions["predicted_pd_12m"]
    elif stage2_bundle.get("modeling_type") == "hazard":
        active_mask = raw_subset[STATUS_COLUMN].isin(ACTIVE_LOAN_STATUSES)
        start_month_column = "months_on_book" if bool(active_mask.all()) else None
        predictions = score_trained_model(
            stage2_bundle,
            raw_subset,
            start_month_column=start_month_column,
        )
        if "predicted_pd_12m" in predictions.columns:
            predictions["predicted_pd"] = predictions["predicted_pd_12m"]
    else:
        feature_subset = build_feature_dataset_from_raw(raw_subset)
        predictions = score_trained_model(stage2_bundle, feature_subset)
    keep_columns = [
        column
        for column in [
            ID_COLUMN,
            "predicted_pd",
            "predicted_pd_12m",
            "predicted_pd_lifetime",
            "model_name",
        ]
        if column in predictions.columns
    ]
    return predictions[keep_columns].copy()


def _month_diff_to_timestamp(start_dates, end_timestamp):
    start = pd.to_datetime(start_dates, errors="coerce").dt.to_period("M")
    end = pd.Timestamp(end_timestamp).to_period("M")
    return (
        (end.year - start.dt.year) * 12
        + (end.month - start.dt.month)
    ).clip(lower=0)
