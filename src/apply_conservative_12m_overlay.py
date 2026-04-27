from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from loss_preprocess import (
    ACTIVE_LOAN_STATUSES,
    RESOLVED_LOAN_STATUSES,
    SPLIT_LABEL_COLUMN,
    STATUS_COLUMN,
    load_loss_raw_dataset,
    prepare_loss_workflow_dataset,
)
from preprocess import ISSUE_DATE_COLUMN, ensure_sample_id
from run_hazard_pd_pipeline import DATA_DIR, PROCESSED_DIR, REPORT_DIR, build_loss_config
from loss_workflow import run_loss_reserve_workflow
from visualization import export_loss_reserve_visuals


RANDOM_STATE = 42
OVERLAY_BUFFER = 1.10
LGD_DENOMINATOR = 0.902
MAX_OVERLAY_PD = 0.95

VALIDATION_BACKTEST_PATH = PROCESSED_DIR / "validation_12m_observable_backtest.csv"
TEST_BACKTEST_PATH = PROCESSED_DIR / "test_12m_observable_backtest.csv"
TEST_PD_PATH = PROCESSED_DIR / "test_with_pd_best_model.csv"
STAGE2_PREDICTION_PATH = PROCESSED_DIR / "dual_pd_stage2_predictions.csv"
STAGE2_CONFIG_PATH = PROCESSED_DIR / "stage2_champion_config.json"
OBSERVABLE_12M_METRICS_PATH = PROCESSED_DIR / "observable_12m_loss_metrics.csv"
OVERLAY_SUMMARY_PATH = PROCESSED_DIR / "conservative_12m_overlay_summary.json"


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _actual_12m_loss(frame: pd.DataFrame) -> pd.Series:
    return np.where(
        frame["actual_default_12m"].fillna(0).astype(int).eq(1),
        pd.to_numeric(frame["actual_net_loss"], errors="coerce").fillna(0).clip(lower=0),
        0.0,
    )


def _required_pd_table(frame: pd.DataFrame, axis: str) -> dict:
    work = frame.copy()
    work["actual_12m_loss"] = _actual_12m_loss(work)
    work["denom"] = (
        pd.to_numeric(work["funded_amnt"], errors="coerce").fillna(0).clip(lower=0)
        * LGD_DENOMINATOR
    )
    grouped = work.groupby(axis, dropna=False).agg(
        rows=("sample_id", "size"),
        actual_12m_loss=("actual_12m_loss", "sum"),
        denominator=("denom", "sum"),
        actual_default_12m=("actual_default_12m", "mean"),
    )
    grouped["required_pd"] = (
        grouped["actual_12m_loss"] / grouped["denominator"].replace(0, np.nan)
    ).clip(lower=0, upper=MAX_OVERLAY_PD)
    return {
        str(key): {
            "rows": int(row["rows"]),
            "actual_default_12m": float(row["actual_default_12m"]),
            "actual_12m_loss": float(row["actual_12m_loss"]),
            "denominator": float(row["denominator"]),
            "required_pd": float(row["required_pd"]) if pd.notna(row["required_pd"]) else np.nan,
            "conservative_floor_pd": float(min(row["required_pd"] * OVERLAY_BUFFER, MAX_OVERLAY_PD))
            if pd.notna(row["required_pd"])
            else np.nan,
        }
        for key, row in grouped.iterrows()
    }


def fit_overlay(validation_frame: pd.DataFrame) -> dict:
    work = validation_frame.copy()
    work["actual_12m_loss"] = _actual_12m_loss(work)
    work["denom"] = (
        pd.to_numeric(work["funded_amnt"], errors="coerce").fillna(0).clip(lower=0)
        * LGD_DENOMINATOR
    )
    portfolio_required_pd = float(work["actual_12m_loss"].sum() / work["denom"].sum())
    return {
        "method": "validation_grade_term_conservative_floor",
        "buffer": OVERLAY_BUFFER,
        "lgd_denominator_assumption": LGD_DENOMINATOR,
        "cap": "static_lifetime_pd",
        "max_overlay_pd": MAX_OVERLAY_PD,
        "portfolio_required_pd": portfolio_required_pd,
        "portfolio_floor_pd": min(portfolio_required_pd * OVERLAY_BUFFER, MAX_OVERLAY_PD),
        "grade_floors": _required_pd_table(work, "grade"),
        "term_floors": _required_pd_table(work, "term_months"),
        "source": str(VALIDATION_BACKTEST_PATH),
    }


def _map_floor(series: pd.Series, table: dict, fallback: float) -> pd.Series:
    mapping = {
        key: value.get("conservative_floor_pd", np.nan)
        for key, value in table.items()
    }
    mapped = series.astype(str).map(mapping)
    return mapped.fillna(fallback).astype(float)


def apply_overlay(
    frame: pd.DataFrame,
    overlay: dict,
    pd_col: str,
    lifetime_col: str,
    *,
    zero_remaining_for_active: bool = False,
) -> pd.DataFrame:
    out = frame.copy()
    prior_col = f"{pd_col}_model_calibrated"
    if prior_col not in out.columns:
        out[prior_col] = out[pd_col]
    fallback = float(overlay["portfolio_floor_pd"])
    grade_floor = _map_floor(out["grade"], overlay["grade_floors"], fallback)
    term_floor = _map_floor(out["term_months"], overlay["term_floors"], fallback)
    floor = np.maximum(grade_floor, term_floor).clip(0, MAX_OVERLAY_PD)
    conservative = np.maximum(
        pd.to_numeric(out[pd_col], errors="coerce").fillna(0).clip(0, 1),
        floor,
    )
    conservative = np.minimum(
        conservative,
        pd.to_numeric(out[lifetime_col], errors="coerce").fillna(1).clip(0, 1),
    )
    if zero_remaining_for_active and "remaining_term" in out.columns:
        remaining = pd.to_numeric(out["remaining_term"], errors="coerce").fillna(0)
        conservative = np.where(remaining.le(0), 0.0, conservative)
        floor = np.where(remaining.le(0), 0.0, floor)
    out["pd_12m_conservative_floor"] = floor
    out[pd_col] = np.asarray(conservative, dtype=float).clip(0, 1)
    return out


def _load_workflow_df() -> pd.DataFrame:
    path = PROCESSED_DIR / "loss_workflow_dataset.csv"
    if path.exists():
        return pd.read_csv(path)
    raw = ensure_sample_id(load_loss_raw_dataset(DATA_DIR / "raw" / "accepted_2007_to_2018Q4.csv"))
    return prepare_loss_workflow_dataset(raw)


def _scope_metadata(workflow_df: pd.DataFrame) -> pd.DataFrame:
    common = [
        "sample_id",
        "grade",
        "term_months",
        "remaining_term",
        ISSUE_DATE_COLUMN,
        SPLIT_LABEL_COLUMN,
        STATUS_COLUMN,
    ]
    resolved = workflow_df[
        workflow_df[SPLIT_LABEL_COLUMN].eq("test")
        & workflow_df[STATUS_COLUMN].isin(RESOLVED_LOAN_STATUSES)
    ][common].copy()
    resolved["prediction_scope"] = "resolved_test"
    active = workflow_df[workflow_df[STATUS_COLUMN].isin(ACTIVE_LOAN_STATUSES)][common].copy()
    active["prediction_scope"] = "active_snapshot"
    observable = workflow_df[
        workflow_df[SPLIT_LABEL_COLUMN].eq("test")
        & (pd.to_datetime(workflow_df[ISSUE_DATE_COLUMN]) < pd.Timestamp("2018-01-01"))
    ][common].copy()
    observable["prediction_scope"] = "observable_12m_test"
    return pd.concat([resolved, active, observable], ignore_index=True)


def refresh_stage2_predictions(overlay: dict, workflow_df: pd.DataFrame) -> pd.DataFrame:
    pred = pd.read_csv(STAGE2_PREDICTION_PATH)
    metadata = _scope_metadata(workflow_df)
    pred = pred.merge(metadata, on=["sample_id", "prediction_scope"], how="left")
    pred = apply_overlay(
        pred,
        overlay,
        pd_col="predicted_pd_12m",
        lifetime_col="predicted_pd_lifetime",
        zero_remaining_for_active=True,
    )
    drop_cols = [
        "grade",
        "term_months",
        "remaining_term",
        ISSUE_DATE_COLUMN,
        SPLIT_LABEL_COLUMN,
        STATUS_COLUMN,
    ]
    pred.drop(columns=[c for c in drop_cols if c in pred.columns], inplace=True)
    pred.to_csv(STAGE2_PREDICTION_PATH, index=False)
    return pred


def refresh_backtests_and_pd_export(overlay: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    validation = pd.read_csv(VALIDATION_BACKTEST_PATH)
    validation = apply_overlay(
        validation,
        overlay,
        pd_col="predicted_pd_12m",
        lifetime_col="predicted_pd_lifetime",
    )
    validation.to_csv(VALIDATION_BACKTEST_PATH, index=False)

    test = pd.read_csv(TEST_BACKTEST_PATH)
    test = apply_overlay(
        test,
        overlay,
        pd_col="predicted_pd_12m",
        lifetime_col="predicted_pd_lifetime",
    )
    test.to_csv(TEST_BACKTEST_PATH, index=False)

    test_pd = pd.read_csv(TEST_PD_PATH)
    replacement = test[["sample_id", "predicted_pd_12m", "pd_12m_conservative_floor"]].copy()
    if "predicted_pd_12m_model_calibrated" in test.columns:
        replacement["predicted_pd_12m_model_calibrated"] = test[
            "predicted_pd_12m_model_calibrated"
        ]
    test_pd = test_pd.drop(
        columns=[
            c
            for c in [
                "predicted_pd_12m",
                "pd_12m_conservative_floor",
                "predicted_pd_12m_model_calibrated",
            ]
            if c in test_pd.columns
        ]
    ).merge(replacement, on="sample_id", how="left")
    test_pd.to_csv(TEST_PD_PATH, index=False)
    return validation, test


def update_stage2_config(overlay: dict) -> dict:
    config = json.loads(STAGE2_CONFIG_PATH.read_text(encoding="utf-8"))
    config.setdefault("hazard_12m_model", {})["pd_12m_conservative_overlay"] = overlay
    config["conservative_12m_overlay_enabled"] = True
    config["conservative_12m_overlay_summary_path"] = str(OVERLAY_SUMMARY_PATH)
    STAGE2_CONFIG_PATH.write_text(json.dumps(config, indent=2, default=_json_default), encoding="utf-8")
    return config


def summarize_outputs(overlay: dict, stage2: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame) -> dict:
    def backtest_summary(frame: pd.DataFrame) -> dict:
        actual_loss = float(_actual_12m_loss(frame).sum())
        denom = float(
            (
                pd.to_numeric(frame["funded_amnt"], errors="coerce").fillna(0).clip(lower=0)
                * LGD_DENOMINATOR
            ).sum()
        )
        return {
            "rows": int(len(frame)),
            "actual_default_12m": float(frame["actual_default_12m"].mean()),
            "actual_12m_loss": actual_loss,
            "mean_pd_12m": float(frame["predicted_pd_12m"].mean()),
            "mean_pd_12m_before_overlay": float(frame["predicted_pd_12m_model_calibrated"].mean()),
            "loss_implied_required_pd": actual_loss / denom if denom else np.nan,
        }

    return {
        "overlay": overlay,
        "stage2_scope_mean_pd": stage2.groupby("prediction_scope")["predicted_pd_12m"].mean().to_dict(),
        "validation": backtest_summary(validation),
        "test": backtest_summary(test),
    }


def main() -> None:
    validation = pd.read_csv(VALIDATION_BACKTEST_PATH)
    overlay = fit_overlay(validation)
    workflow_df = _load_workflow_df()
    stage2 = refresh_stage2_predictions(overlay, workflow_df)
    validation, test = refresh_backtests_and_pd_export(overlay)
    config = update_stage2_config(overlay)
    summary = summarize_outputs(overlay, stage2, validation, test)
    _write_json(OVERLAY_SUMMARY_PATH, summary)

    loss_config = build_loss_config(config)
    loss_config["observable_12m_loss_metrics_path"] = str(OBSERVABLE_12M_METRICS_PATH)
    loss_result = run_loss_reserve_workflow(loss_config)
    export_loss_reserve_visuals(
        loss_result,
        output_dir=REPORT_DIR / "figures" / "loss_reserve",
        show=False,
    )
    portfolio = pd.read_csv(PROCESSED_DIR / "portfolio_expected_loss_summary.csv")
    print(json.dumps({
        "overlay_summary_path": str(OVERLAY_SUMMARY_PATH),
        "portfolio_summary": portfolio.to_dict(orient="records"),
    }, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
