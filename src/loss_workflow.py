from pathlib import Path

import numpy as np
import pandas as pd

from experiment import fit_locked_model, score_trained_model
from hazard import DEFAULT_HAZARD_PARAM_GRID, fit_pooled_logistic_hazard, score_hazard_probabilities
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
from preprocess import ID_COLUMN, build_feature_dataset_from_raw, load_processed_data


def run_loss_reserve_workflow(config):
    raw_data_path = Path(config["raw_data_path"])
    pd_prediction_path = Path(config["pd_prediction_path"])
    loss_workflow_path = Path(config["loss_workflow_path"])
    charged_off_proxy_path = Path(config["charged_off_proxy_path"])
    cecl_snapshot_path = Path(config["cecl_snapshot_path"])
    test_loss_metrics_path = Path(config["test_loss_metrics_path"])
    segment_summary_path = Path(config["segment_summary_path"])
    portfolio_summary_path = Path(config["portfolio_summary_path"])
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

    hazard_result = fit_pooled_logistic_hazard(
        workflow_df,
        param_grid=config.get("hazard_param_grid", DEFAULT_HAZARD_PARAM_GRID),
    )

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

    stage2_benchmark_summary = {}
    stage2_scoring_config = config.get("stage2_scoring", {})
    if stage2_scoring_config:
        stage2_bundle = fit_locked_model(
            stage2_scoring_config,
            load_processed_data(stage2_scoring_config["processed_data_path"]),
            model_name=stage2_scoring_config.get("selected_model_key"),
        )
        resolved_stage2_predictions = _score_stage2_subset(
            stage2_bundle,
            raw_loss_df,
            resolved_holdout["sample_id"],
        )
        active_stage2_predictions = _score_stage2_subset(
            stage2_bundle,
            raw_loss_df,
            active_snapshot["sample_id"],
        )
        resolved_holdout = resolved_holdout.merge(
            resolved_stage2_predictions.rename(
                columns={"predicted_pd": "stage2_champion_pd", "model_name": "stage2_model_name"}
            ),
            on="sample_id",
            how="left",
        )
        active_snapshot = active_snapshot.merge(
            active_stage2_predictions.rename(
                columns={"predicted_pd": "stage2_champion_pd", "model_name": "stage2_model_name"}
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
            "prediction_source": "locked_stage2_model",
            "used_for_12m_el": True,
        }
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
        frame["pd_12m_fixed_horizon"] = frame["stage2_champion_pd"].fillna(
            frame["pd_12m_hazard"]
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

    _write_csv(resolved_holdout, test_loss_metrics_path)
    _write_csv(active_snapshot, cecl_snapshot_path)

    portfolio_summary = pd.concat(
        [
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
        "hazard_validation_overall": _build_hazard_metric_table(
            hazard_result["validation_metrics"],
            hazard_result["validation_metrics_12m"],
            split_name="validation",
        ),
        "hazard_test_overall": _build_hazard_metric_table(
            hazard_result["test_metrics"],
            hazard_result["test_metrics_12m"],
            split_name="test",
        ),
        "hazard_validation_vintage": hazard_result["validation_vintage_table"],
        "hazard_test_vintage": hazard_result["test_vintage_table"],
        "portfolio_summary": portfolio_summary,
        "segment_summary": segment_summary,
        "resolved_holdout_sample": resolved_holdout[
            [
                "sample_id",
                "issue_date",
                STATUS_COLUMN,
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
        "active_snapshot_sample": active_snapshot[
            [
                "sample_id",
                "issue_date",
                STATUS_COLUMN,
                "remaining_term",
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
    summary["actual_loss_amount"] = (
        float(df["actual_net_loss"].sum())
        if include_actual_loss and "actual_net_loss" in df.columns
        else np.nan
    )
    summary["actual_loss_rate"] = (
        float(df["actual_net_loss"].sum() / df["funded_amnt"].sum())
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


def _build_hazard_metric_table(lifetime_metrics, metrics_12m, split_name):
    return pd.DataFrame(
        [
            {
                "split_name": split_name,
                "pd_measure": "lifetime",
                "auc": float(lifetime_metrics["AUC"]),
                "ks": float(lifetime_metrics["KS"]),
                "brier": float(lifetime_metrics["Brier"]),
            },
            {
                "split_name": split_name,
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

    raw_subset = raw_loss_df.loc[sample_ids].copy()
    raw_subset[ID_COLUMN] = raw_subset[ID_COLUMN].astype(int)
    feature_subset = build_feature_dataset_from_raw(raw_subset)
    predictions = score_trained_model(stage2_bundle, feature_subset)
    keep_columns = [column for column in [ID_COLUMN, "predicted_pd", "model_name"] if column in predictions.columns]
    return predictions[keep_columns].copy()
