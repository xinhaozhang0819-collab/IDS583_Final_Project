from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from calendar_bigdata import read_modeling_sample
from evaluation import evaluate_model
from experiment import (
    COMPACT_CALENDAR_HAZARD_PROFILE,
    FULL_CALENDAR_HAZARD_PROFILE,
    ID_COLUMN,
    MODEL_LABELS,
    SNAPSHOT_MONTH_COLUMN,
    _align_feature_frame,
    _build_hazard_design_matrix,
    _build_model,
    _safe_evaluate_model,
)
from loss_preprocess import (
    ACTIVE_LOAN_STATUSES,
    RESOLVED_LOAN_STATUSES,
    SPLIT_LABEL_COLUMN,
    STATUS_COLUMN,
    apply_expected_lgd_lookup,
    build_charged_off_loss_proxy,
    fit_expected_lgd_lookup,
    load_loss_raw_dataset,
    prepare_loss_workflow_dataset,
)
from preprocess import ISSUE_DATE_COLUMN, ensure_sample_id
from run_dual_pd_30min_training import (
    DATA_CUTOFF,
    OBSERVABLE_12M_METRICS_PATH,
    STAGE2_CONFIG_PATH,
    STAGE2_PREDICTION_PATH,
    STATIC_HGB_PARAMS,
    TEST_12M_BACKTEST_PATH,
    TEST_PD_PATH,
    VALIDATION_12M_BACKTEST_PATH,
    _fit_static_lifetime_model,
    _prepare_hazard_scoring_panel,
    _scope_raw,
    _score_static_lifetime,
)
from run_hazard_pd_pipeline import DATA_DIR, PROCESSED_DIR, REPORT_DIR, build_loss_config
from loss_workflow import run_loss_reserve_workflow
from visualization import export_loss_reserve_visuals


RANDOM_STATE = 42
TRAIN_CAP = 300_000
VALIDATION_CAP = 150_000
TEST_CAP = 200_000
TRAINING_BUDGET_SECONDS = 30 * 60
DIRECT_MODEL_SOURCE = "direct_active_snapshot_12m_xgboost"
DIRECT_RUNTIME_PATH = PROCESSED_DIR / "direct_active_12m_runtime.json"
DIRECT_SEARCH_PATH = PROCESSED_DIR / "direct_active_12m_search_results.csv"
DIRECT_BACKTEST_PATH = PROCESSED_DIR / "direct_active_12m_backtest.csv"
DIRECT_SAMPLE_PATH = PROCESSED_DIR / "direct_active_12m_training_view.parquet"
DIRECT_SUMMARY_PATH = PROCESSED_DIR / "direct_active_12m_training_view_summary.json"
HAZARD_STAGE2_BACKUP_PATH = PROCESSED_DIR / "dual_pd_stage2_predictions.csv"

DIRECT_XGB_CANDIDATES = [
    {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.08},
    {"n_estimators": 120, "max_depth": 4, "learning_rate": 0.06},
    {"n_estimators": 120, "max_depth": 5, "learning_rate": 0.06},
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
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    return str(value)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _logit_shift(probabilities, shift, eps=1e-6):
    p = np.clip(np.asarray(probabilities, dtype=float), eps, 1.0 - eps)
    logits = np.log(p / (1.0 - p)) + float(shift)
    logits = np.clip(logits, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-logits))


def _fit_loss_weighted_intercept(probabilities, denominator, target_loss):
    denominator = np.asarray(denominator, dtype=float)
    target_mean = float(np.clip(target_loss / max(denominator.sum(), 1e-9), 1e-6, 1.0 - 1e-6))
    low, high = -50.0, 50.0
    for _ in range(100):
        mid = (low + high) / 2.0
        shifted = _logit_shift(probabilities, mid)
        weighted_mean = float(np.sum(shifted * denominator) / max(denominator.sum(), 1e-9))
        if weighted_mean < target_mean:
            low = mid
        else:
            high = mid
    return float((low + high) / 2.0), target_mean


def _candidate_plan():
    xgb_base = {
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_lambda": 1.0,
        "random_state": RANDOM_STATE,
        "n_jobs": 5,
    }
    return [
        {
            "model_key": "logistic_regression",
            "candidate_id": 1,
            "params": {"C": 1.0, "max_iter": 300, "class_weight": None},
            "feature_profile": COMPACT_CALENDAR_HAZARD_PROFILE,
            "champion_eligible": False,
        },
        {
            "model_key": "random_forest",
            "candidate_id": 1,
            "params": {
                "n_estimators": 80,
                "max_depth": 10,
                "min_samples_split": 200,
                "min_samples_leaf": 60,
                "max_features": "sqrt",
                "bootstrap": True,
                "max_samples": 0.7,
                "n_jobs": -1,
                "random_state": RANDOM_STATE,
            },
            "feature_profile": FULL_CALENDAR_HAZARD_PROFILE,
            "champion_eligible": False,
        },
        *[
            {
                "model_key": "xgboost",
                "candidate_id": index + 1,
                "params": {**copy.deepcopy(xgb_base), **override},
                "feature_profile": FULL_CALENDAR_HAZARD_PROFILE,
                "champion_eligible": True,
            }
            for index, override in enumerate(DIRECT_XGB_CANDIDATES)
        ],
    ]


def _actual_12m_loss(frame: pd.DataFrame) -> pd.Series:
    if "actual_net_loss" not in frame.columns:
        return pd.Series(0.0, index=frame.index)
    return np.where(
        frame["target_12m"].fillna(0).astype(int).eq(1),
        pd.to_numeric(frame["actual_net_loss"], errors="coerce").fillna(0).clip(lower=0),
        0.0,
    )


def _enrich_panel_with_workflow_fields(panel: pd.DataFrame, workflow_df: pd.DataFrame) -> pd.DataFrame:
    merge_columns = [
        ID_COLUMN,
        "actual_net_loss",
        "ead_current",
    ]
    available = [column for column in merge_columns if column in workflow_df.columns]
    if len(available) <= 1:
        return panel.copy()
    enrichment = workflow_df[available].drop_duplicates(subset=[ID_COLUMN])
    frame = panel.copy()
    duplicate_payload = [column for column in available if column != ID_COLUMN and column in frame.columns]
    if duplicate_payload:
        frame = frame.drop(columns=duplicate_payload)
    return frame.merge(enrichment, on=ID_COLUMN, how="left")


def _prepare_direct_frame(panel: pd.DataFrame, lgd_lookup: dict) -> pd.DataFrame:
    frame = panel.copy()
    frame[SNAPSHOT_MONTH_COLUMN] = pd.to_datetime(frame[SNAPSHOT_MONTH_COLUMN], errors="coerce")
    frame = frame[frame["split_label"].isin(["train", "validation", "test", "scoring"])].copy()
    supervised = frame["split_label"].isin(["train", "validation", "test"])
    frame = frame[
        (~supervised)
        | (
            frame[SNAPSHOT_MONTH_COLUMN].le(pd.Timestamp("2017-12-01"))
            & frame["target_12m"].notna()
        )
    ].copy()
    if "issue_year" not in frame.columns:
        frame["issue_year"] = pd.to_datetime(
            frame[ISSUE_DATE_COLUMN], errors="coerce"
        ).dt.year.astype("Int64")
    frame = apply_expected_lgd_lookup(frame, lgd_lookup)
    if "actual_net_loss" not in frame.columns:
        frame["actual_net_loss"] = 0.0
    if "ead_current" not in frame.columns:
        frame["ead_current"] = 0.0
    frame["ead_snapshot"] = np.where(
        frame["split_label"].eq("scoring"),
        pd.to_numeric(frame["ead_current"], errors="coerce").fillna(0),
        pd.to_numeric(frame["scheduled_balance_proxy"], errors="coerce").fillna(0),
    )
    frame["ead_snapshot"] = frame["ead_snapshot"].clip(lower=0)
    frame["el_weight"] = (
        frame["expected_lgd"].fillna(0).clip(0, 1) * frame["ead_snapshot"].fillna(0)
    ).clip(lower=1.0)
    frame["direct_12m_actual_loss"] = _actual_12m_loss(frame)
    frame["direct_12m_denominator"] = (
        frame["expected_lgd"].fillna(0).clip(0, 1) * frame["ead_snapshot"].fillna(0)
    )
    return frame.sort_values(["split_label", SNAPSHOT_MONTH_COLUMN, ID_COLUMN]).reset_index(drop=True)


def _sample_split(
    partition: pd.DataFrame,
    cap: int,
    random_state: int,
    preserve_positive: bool = False,
) -> pd.DataFrame:
    if len(partition) <= cap:
        return partition.copy()
    y = partition["target_12m"].fillna(0).astype(int)
    positives = partition[y.eq(1)]
    negatives = partition[y.eq(0)]
    if preserve_positive:
        pos_cap = min(len(positives), int(cap * 0.6))
    else:
        pos_cap = min(len(positives), int(round(cap * len(positives) / max(len(partition), 1))))
    positive_sample = (
        positives.sample(n=pos_cap, random_state=random_state)
        if len(positives) > pos_cap
        else positives
    )
    neg_cap = max(0, cap - len(positive_sample))
    negative_sample = negatives.sample(n=min(len(negatives), neg_cap), random_state=random_state + 1)
    return pd.concat([positive_sample, negative_sample], ignore_index=True)


def _build_direct_training_view(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    parts = [
        _sample_split(
            frame[frame["split_label"].eq("train")],
            TRAIN_CAP,
            RANDOM_STATE,
            preserve_positive=True,
        ),
        _sample_split(frame[frame["split_label"].eq("validation")], VALIDATION_CAP, RANDOM_STATE + 17),
        _sample_split(frame[frame["split_label"].eq("test")], TEST_CAP, RANDOM_STATE + 31),
        frame[frame["split_label"].eq("scoring")].copy(),
    ]
    view = pd.concat(parts, ignore_index=True).sort_values(
        ["split_label", SNAPSHOT_MONTH_COLUMN, ID_COLUMN]
    ).reset_index(drop=True)
    summary = (
        view.groupby("split_label", dropna=False)
        .agg(
            rows=(ID_COLUMN, "size"),
            loans=(ID_COLUMN, "nunique"),
            target_12m=("target_12m", "sum"),
            target_12m_rate=("target_12m", "mean"),
            min_snapshot=(SNAPSHOT_MONTH_COLUMN, "min"),
            max_snapshot=(SNAPSHOT_MONTH_COLUMN, "max"),
        )
        .reset_index()
    )
    payload = {
        "view_path": str(DIRECT_SAMPLE_PATH),
        "caps": {"train": TRAIN_CAP, "validation": VALIDATION_CAP, "test": TEST_CAP},
        "sample_id_snapshot_unique": bool(
            not view.duplicated([ID_COLUMN, SNAPSHOT_MONTH_COLUMN]).any()
        ),
        "summary": summary.to_dict(orient="records"),
    }
    return view, payload


def _design(frame: pd.DataFrame, feature_profile: str, columns=None):
    return _build_hazard_design_matrix(
        frame,
        feature_profile=feature_profile,
        design_columns=columns,
    )


def _fit_model(model_key, params, x_train, y_train, weights):
    model = _build_model(model_key, params)
    if model_key == "logistic_regression":
        model.fit(x_train, y_train, classifier__sample_weight=weights)
    else:
        model.fit(x_train, y_train, sample_weight=weights)
    return model


def _predict(model, x):
    return model.predict_proba(x)[:, 1].clip(0, 1)


def _segment_floor_table(validation: pd.DataFrame, axis: str, buffer: float = 1.10) -> dict:
    grouped = validation.groupby(axis, dropna=False).agg(
        rows=(ID_COLUMN, "size"),
        actual_loss=("direct_12m_actual_loss", "sum"),
        denominator=("direct_12m_denominator", "sum"),
        target_rate=("target_12m", "mean"),
    )
    grouped["required_pd"] = grouped["actual_loss"] / grouped["denominator"].replace(0, np.nan)
    grouped["conservative_floor_pd"] = (grouped["required_pd"] * buffer).clip(lower=0, upper=0.95)
    return {
        str(key): {
            "rows": int(row["rows"]),
            "target_rate": float(row["target_rate"]),
            "actual_loss": float(row["actual_loss"]),
            "denominator": float(row["denominator"]),
            "required_pd": float(row["required_pd"]) if pd.notna(row["required_pd"]) else np.nan,
            "conservative_floor_pd": float(row["conservative_floor_pd"])
            if pd.notna(row["conservative_floor_pd"])
            else np.nan,
        }
        for key, row in grouped.iterrows()
    }


def _map_floor(series: pd.Series, floor_table: dict, fallback: float) -> pd.Series:
    mapping = {key: value["conservative_floor_pd"] for key, value in floor_table.items()}
    return series.astype(str).map(mapping).fillna(fallback).astype(float)


def _build_workflow_12m_calibration_reference(
    workflow_df: pd.DataFrame,
    lgd_lookup: dict,
) -> pd.DataFrame:
    reference = workflow_df[
        workflow_df[SPLIT_LABEL_COLUMN].eq("validation")
        & pd.to_datetime(workflow_df[ISSUE_DATE_COLUMN], errors="coerce").lt(pd.Timestamp("2017-01-01"))
    ].copy()
    reference = apply_expected_lgd_lookup(reference, lgd_lookup)
    reference["target_12m"] = reference["actual_default_12m"].fillna(0).astype(int)
    reference["direct_12m_actual_loss"] = (
        reference["target_12m"]
        * pd.to_numeric(reference["actual_net_loss"], errors="coerce").fillna(0).clip(lower=0)
    )
    reference["direct_12m_denominator"] = (
        reference["expected_lgd"].fillna(0).clip(0, 1)
        * pd.to_numeric(reference["funded_amnt"], errors="coerce").fillna(0).clip(lower=0)
    )
    return reference


def _fit_direct_calibrator(
    validation: pd.DataFrame,
    raw_pd: np.ndarray,
    buffer: float = 1.10,
    coverage_reference: pd.DataFrame | None = None,
) -> dict:
    target_frame = coverage_reference if coverage_reference is not None else validation
    reference_actual_loss = float(target_frame["direct_12m_actual_loss"].sum())
    reference_denominator = float(target_frame["direct_12m_denominator"].sum())
    target_mean_reference = float(
        np.clip(reference_actual_loss * buffer / max(reference_denominator, 1e-9), 1e-6, 0.95)
    )
    target_loss = float(target_mean_reference * validation["direct_12m_denominator"].sum())
    shift, target_mean = _fit_loss_weighted_intercept(
        raw_pd,
        validation["direct_12m_denominator"].fillna(0).to_numpy(),
        target_loss,
    )
    denominator = float(validation["direct_12m_denominator"].sum())
    actual_loss = float(validation["direct_12m_actual_loss"].sum())
    portfolio_required_pd = actual_loss / denominator if denominator else np.nan
    return {
        "method": "loss_weighted_logit_intercept_plus_grade_term_floor",
        "intercept_shift": shift,
        "buffer": buffer,
        "target_loss": target_loss,
        "target_weighted_mean_pd": target_mean,
        "coverage_reference": "workflow_validation_observable_12m"
        if coverage_reference is not None
        else "direct_training_validation_sample",
        "reference_rows": int(len(target_frame)),
        "reference_actual_loss": reference_actual_loss,
        "reference_denominator": reference_denominator,
        "reference_required_pd": reference_actual_loss / reference_denominator
        if reference_denominator
        else np.nan,
        "reference_target_pd_with_buffer": target_mean_reference,
        "validation_actual_loss": actual_loss,
        "validation_denominator": denominator,
        "validation_portfolio_required_pd": portfolio_required_pd,
        "grade_floors": _segment_floor_table(target_frame, "grade", buffer=buffer),
        "term_floors": _segment_floor_table(target_frame, "term_months", buffer=buffer),
        "cap": "static_lifetime_pd",
    }


def _apply_direct_calibrator(frame: pd.DataFrame, raw_pd: np.ndarray, calibrator: dict) -> pd.DataFrame:
    out = frame.copy()
    calibrated = _logit_shift(raw_pd, calibrator.get("intercept_shift", 0.0))
    fallback = float(calibrator.get("target_weighted_mean_pd", 0.0))
    grade_floor = _map_floor(out["grade"], calibrator.get("grade_floors", {}), fallback)
    term_floor = _map_floor(out["term_months"], calibrator.get("term_floors", {}), fallback)
    floor = np.maximum(grade_floor, term_floor).clip(0, 0.95)
    final = np.maximum(calibrated, floor)
    if "predicted_pd_lifetime" in out.columns:
        final = np.minimum(final, pd.to_numeric(out["predicted_pd_lifetime"], errors="coerce").fillna(1).clip(0, 1))
    out["predicted_pd_12m_direct_raw"] = raw_pd
    out["predicted_pd_12m_direct_calibrated"] = calibrated
    out["pd_12m_direct_conservative_floor"] = floor
    out["predicted_pd_12m_direct"] = np.asarray(final, dtype=float).clip(0, 1)
    return out


def _coverage_metrics(frame: pd.DataFrame, pd_column: str) -> dict:
    expected = float((frame[pd_column] * frame["direct_12m_denominator"]).sum())
    actual = float(frame["direct_12m_actual_loss"].sum())
    by_grade = {}
    for grade in ["D", "E", "F", "G"]:
        part = frame[frame["grade"].astype(str).eq(grade)]
        actual_g = float(part["direct_12m_actual_loss"].sum())
        expected_g = float((part[pd_column] * part["direct_12m_denominator"]).sum())
        by_grade[grade] = expected_g / actual_g if actual_g else np.nan
    term60 = frame[pd.to_numeric(frame["term_months"], errors="coerce").eq(60)]
    actual60 = float(term60["direct_12m_actual_loss"].sum())
    expected60 = float((term60[pd_column] * term60["direct_12m_denominator"]).sum())
    return {
        "expected_loss": expected,
        "actual_loss": actual,
        "coverage": expected / actual if actual else np.nan,
        "grade_d_to_g_min_coverage": float(np.nanmin(list(by_grade.values()))),
        "grade_coverage": by_grade,
        "term60_coverage": expected60 / actual60 if actual60 else np.nan,
    }


def _fit_direct_model(
    view: pd.DataFrame,
    calibration_reference: pd.DataFrame | None = None,
) -> tuple[dict, pd.DataFrame, dict]:
    train = view[view["split_label"].eq("train")].copy()
    validation = view[view["split_label"].eq("validation")].copy()
    candidates = []
    bundles = {}

    for candidate in _candidate_plan():
        start = time.perf_counter()
        x_train = _design(train, candidate["feature_profile"])
        y_train = train["target_12m"].fillna(0).astype(int)
        weights = train["el_weight"].fillna(1).astype(float)
        model = _fit_model(candidate["model_key"], candidate["params"], x_train, y_train, weights)
        feature_columns = list(x_train.columns)
        fit_seconds = time.perf_counter() - start

        score_start = time.perf_counter()
        x_validation = _design(validation, candidate["feature_profile"], columns=feature_columns)
        raw_pd = _predict(model, x_validation)
        calibrator = _fit_direct_calibrator(
            validation,
            raw_pd,
            coverage_reference=calibration_reference,
        )
        validation_scored = _apply_direct_calibrator(validation, raw_pd, calibrator)
        metrics = _safe_evaluate_model(
            validation["target_12m"].fillna(0).astype(int),
            validation_scored["predicted_pd_12m_direct"],
        )
        coverage = _coverage_metrics(validation_scored, "predicted_pd_12m_direct")
        score_seconds = time.perf_counter() - score_start
        row = {
            "candidate_order": len(candidates) + 1,
            "candidate_id": candidate["candidate_id"],
            "model_key": candidate["model_key"],
            "model_name": MODEL_LABELS[candidate["model_key"]],
            "champion_eligible": candidate["champion_eligible"],
            "feature_profile": candidate["feature_profile"],
            "feature_count": len(feature_columns),
            "params": json.dumps(candidate["params"], sort_keys=True, default=_json_default),
            "validation_auc": float(metrics["AUC"]),
            "validation_ks": float(metrics["KS"]),
            "validation_brier": float(metrics["Brier"]),
            "validation_el_coverage": float(coverage["coverage"]),
            "validation_grade_d_to_g_min_coverage": float(coverage["grade_d_to_g_min_coverage"]),
            "validation_term60_coverage": float(coverage["term60_coverage"]),
            "passes_conservative_constraints": bool(
                candidate["champion_eligible"]
                and coverage["coverage"] >= 1.10
                and coverage["grade_d_to_g_min_coverage"] >= 1.0
                and coverage["term60_coverage"] >= 1.0
            ),
            "fit_seconds": float(fit_seconds),
            "score_seconds": float(score_seconds),
        }
        candidates.append(row)
        key = (candidate["model_key"], candidate["candidate_id"])
        bundles[key] = {
            "model": model,
            "model_key": candidate["model_key"],
            "model_name": MODEL_LABELS[candidate["model_key"]],
            "params": copy.deepcopy(candidate["params"]),
            "feature_profile": candidate["feature_profile"],
            "feature_columns": feature_columns,
            "calibrator": calibrator,
        }
        print(
            f"direct 12M candidate {row['model_name']} #{row['candidate_id']} "
            f"auc={row['validation_auc']:.4f} coverage={row['validation_el_coverage']:.3f} "
            f"fit={fit_seconds:.1f}s",
            flush=True,
        )

    search = pd.DataFrame(candidates)
    eligible = search[search["passes_conservative_constraints"]].copy()
    if eligible.empty:
        eligible = search[search["champion_eligible"]].copy()
    selected = eligible.sort_values(
        ["validation_auc", "validation_ks", "validation_el_coverage"],
        ascending=False,
    ).iloc[0]
    selected_key = (selected["model_key"], int(selected["candidate_id"]))

    train_validation = view[view["split_label"].isin(["train", "validation"])].copy()
    selected_bundle = bundles[selected_key]
    x_train_validation = _design(
        train_validation,
        selected_bundle["feature_profile"],
    )
    final_model = _fit_model(
        selected_bundle["model_key"],
        selected_bundle["params"],
        x_train_validation,
        train_validation["target_12m"].fillna(0).astype(int),
        train_validation["el_weight"].fillna(1).astype(float),
    )
    validation_raw = _predict(
        final_model,
        _design(validation, selected_bundle["feature_profile"], columns=list(x_train_validation.columns)),
    )
    final_calibrator = _fit_direct_calibrator(
        validation,
        validation_raw,
        coverage_reference=calibration_reference,
    )
    final_bundle = {
        **selected_bundle,
        "model": final_model,
        "feature_columns": list(x_train_validation.columns),
        "calibrator": final_calibrator,
        "model_source": DIRECT_MODEL_SOURCE,
    }
    runtime = {
        "selected_model_key": selected_bundle["model_key"],
        "selected_model_name": selected_bundle["model_name"],
        "selected_candidate_id": int(selected["candidate_id"]),
        "selected_params": selected_bundle["params"],
        "selected_validation_auc": float(selected["validation_auc"]),
        "selected_validation_ks": float(selected["validation_ks"]),
        "selected_validation_brier": float(selected["validation_brier"]),
        "selected_validation_el_coverage": float(selected["validation_el_coverage"]),
        "selected_validation_grade_d_to_g_min_coverage": float(
            selected["validation_grade_d_to_g_min_coverage"]
        ),
        "selected_validation_term60_coverage": float(selected["validation_term60_coverage"]),
        "candidate_count": int(len(search)),
        "calibrator": final_calibrator,
    }
    return final_bundle, search, runtime


def _score_direct_frame(bundle: dict, scoring: pd.DataFrame, static_pred: pd.DataFrame | None = None) -> pd.DataFrame:
    frame = scoring.copy().reset_index(drop=True)
    if static_pred is not None:
        frame = frame.merge(
            static_pred[[ID_COLUMN, "predicted_pd_lifetime_static"]],
            on=ID_COLUMN,
            how="left",
        )
        frame["predicted_pd_lifetime"] = frame["predicted_pd_lifetime_static"]
    x = _design(frame, bundle["feature_profile"], columns=bundle["feature_columns"])
    raw_pd = _predict(bundle["model"], x)
    scored = _apply_direct_calibrator(frame, raw_pd, bundle["calibrator"])
    return scored


def _score_direct_scope(static_bundle, direct_bundle, raw_scope_df, scope_name, snapshot_mode):
    static_pred = _score_static_lifetime(static_bundle, raw_scope_df)
    scoring = _prepare_hazard_scoring_panel(direct_bundle, raw_scope_df, snapshot_mode=snapshot_mode)
    scored = _score_direct_frame(direct_bundle, scoring, static_pred=static_pred)
    out = static_pred.merge(
        scored[
            [
                ID_COLUMN,
                "predicted_pd_12m_direct_raw",
                "predicted_pd_12m_direct_calibrated",
                "pd_12m_direct_conservative_floor",
                "predicted_pd_12m_direct",
            ]
        ],
        on=ID_COLUMN,
        how="left",
    )
    out["prediction_scope"] = scope_name
    out["predicted_pd"] = out["predicted_pd_lifetime_static"].clip(0, 1)
    out["predicted_pd_lifetime"] = out["predicted_pd_lifetime_static"].clip(0, 1)
    out["predicted_pd_12m"] = out["predicted_pd_12m_direct"].clip(0, 1)
    out["model_name"] = "Dual PD: Static HGB lifetime + direct active-snapshot 12M"
    out["hazard_12m_model_name"] = "XGBoost Hazard Comparison"
    out["direct_12m_model_name"] = direct_bundle["model_name"]
    out["pd_12m_model_source"] = DIRECT_MODEL_SOURCE
    out["pd_12m_champion_source"] = DIRECT_MODEL_SOURCE
    return out


def _previous_hazard_predictions() -> pd.DataFrame:
    if not HAZARD_STAGE2_BACKUP_PATH.exists():
        return pd.DataFrame()
    previous = pd.read_csv(HAZARD_STAGE2_BACKUP_PATH)
    columns = [
        ID_COLUMN,
        "prediction_scope",
        "predicted_pd_12m_raw",
        "predicted_pd_12m_model_calibrated",
        "predicted_hazard_1m",
        "predicted_pd_12m_hazard_raw",
        "predicted_pd_12m_hazard",
        "predicted_hazard_1m_hazard",
    ]
    keep = [column for column in columns if column in previous.columns]
    previous = previous[keep].copy()
    previous = previous.rename(
        columns={
            "predicted_pd_12m_raw": "predicted_pd_12m_hazard_raw",
            "predicted_pd_12m_model_calibrated": "predicted_pd_12m_hazard",
            "predicted_hazard_1m": "predicted_hazard_1m_hazard",
        }
    )
    previous = previous.loc[:, ~previous.columns.duplicated()].copy()
    return previous


def _build_stage2_predictions(static_bundle, direct_bundle, raw_loss_df, workflow_df):
    resolved = workflow_df[
        workflow_df[SPLIT_LABEL_COLUMN].eq("test") & workflow_df[STATUS_COLUMN].isin(RESOLVED_LOAN_STATUSES)
    ].copy()
    active = workflow_df[workflow_df[STATUS_COLUMN].isin(ACTIVE_LOAN_STATUSES)].copy()
    observable_12m = workflow_df[
        workflow_df[SPLIT_LABEL_COLUMN].eq("test")
        & (pd.to_datetime(workflow_df[ISSUE_DATE_COLUMN]) < pd.Timestamp("2018-01-01"))
    ].copy()
    frames = []
    for scope_name, scope_df, mode in [
        ("resolved_test", resolved, "origination"),
        ("active_snapshot", active, "active_cutoff"),
        ("observable_12m_test", observable_12m, "origination"),
    ]:
        print(f"Scoring direct 12M scope={scope_name} rows={len(scope_df):,}", flush=True)
        frames.append(
            _score_direct_scope(
                static_bundle,
                direct_bundle,
                _scope_raw(raw_loss_df, scope_df),
                scope_name,
                mode,
            )
        )
    predictions = pd.concat(frames, ignore_index=True)
    hazard = _previous_hazard_predictions()
    if not hazard.empty:
        predictions = predictions.merge(hazard, on=[ID_COLUMN, "prediction_scope"], how="left")
    else:
        predictions["predicted_pd_12m_hazard"] = np.nan
        predictions["predicted_pd_12m_hazard_raw"] = np.nan
    return predictions


def _write_test_exports(direct_bundle, view, static_bundle, raw_loss_df, workflow_df):
    test = view[view["split_label"].eq("test")].copy()
    static_pred = _score_static_lifetime(static_bundle, _scope_raw(raw_loss_df, test))
    scored = _score_direct_frame(direct_bundle, test, static_pred=static_pred)
    metrics = _safe_evaluate_model(
        scored["target_12m"].fillna(0).astype(int),
        scored["predicted_pd_12m_direct"],
    )
    coverage = _coverage_metrics(scored, "predicted_pd_12m_direct")
    scored["predicted_pd_12m"] = scored["predicted_pd_12m_direct"]
    scored["predicted_pd"] = scored["predicted_pd_lifetime"]
    scored["pd_12m_model_source"] = DIRECT_MODEL_SOURCE
    scored.to_csv(DIRECT_BACKTEST_PATH, index=False)

    observable = workflow_df[
        workflow_df[SPLIT_LABEL_COLUMN].eq("test")
        & (pd.to_datetime(workflow_df[ISSUE_DATE_COLUMN]) < pd.Timestamp("2018-01-01"))
    ].copy()
    pred = _score_direct_scope(
        static_bundle,
        direct_bundle,
        _scope_raw(raw_loss_df, observable),
        "observable_12m_test",
        "origination",
    )
    export = observable.merge(pred, on=ID_COLUMN, how="left")
    export.to_csv(TEST_12M_BACKTEST_PATH, index=False)
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
        "predicted_pd_12m_direct_raw",
        "predicted_pd_12m_direct_calibrated",
        "predicted_pd_12m_direct",
        "predicted_pd_12m",
        "predicted_pd_lifetime_static_raw",
        "predicted_pd_lifetime_static",
        "predicted_pd",
        "model_name",
        "lifetime_model_name",
        "direct_12m_model_name",
        "pd_12m_model_source",
    ]
    export[[c for c in keep if c in export.columns]].to_csv(TEST_PD_PATH, index=False)
    return {"test_metrics": metrics, "test_coverage": coverage, "test_rows": int(len(scored))}


def _write_config(static_summary, direct_bundle, direct_runtime, view_summary, runtime_seconds):
    existing = {}
    if STAGE2_CONFIG_PATH.exists():
        existing = json.loads(STAGE2_CONFIG_PATH.read_text(encoding="utf-8"))
    config = {
        **existing,
        "modeling_type": "dual_pd_direct_12m",
        "training_mode": "direct_active_snapshot_12m_static_lifetime_hazard_comparison",
        "data_cutoff": DATA_CUTOFF.strftime("%Y-%m-%d"),
        "processed_data_path": str(DATA_DIR / "raw" / "accepted_2007_to_2018Q4.csv"),
        "precomputed_dual_pd_prediction_path": str(STAGE2_PREDICTION_PATH),
        "observable_12m_loss_metrics_path": str(OBSERVABLE_12M_METRICS_PATH),
        "selected_model_key": "dual_pd_static_hgb_direct_active_snapshot_12m",
        "selected_lifetime_model_key": "hist_gradient_boosting",
        "selected_12m_model_key": "direct_12m_xgboost",
        "model_name": "Dual PD: Static HGB lifetime + direct active-snapshot 12M",
        "lifetime_model": static_summary,
        "direct_12m_model": {
            "model_key": direct_runtime["selected_model_key"],
            "model_name": direct_runtime["selected_model_name"],
            "source": DIRECT_MODEL_SOURCE,
            "params": direct_runtime["selected_params"],
            "feature_profile": direct_bundle["feature_profile"],
            "feature_count": len(direct_bundle["feature_columns"]),
            "calibration": direct_bundle["calibrator"],
        },
        "hazard_12m_comparison_model": existing.get("hazard_12m_model", {}),
        "direct_active_12m_training_view_path": str(DIRECT_SAMPLE_PATH),
        "direct_active_12m_backtest_path": str(DIRECT_BACKTEST_PATH),
        "direct_active_12m_search_results_path": str(DIRECT_SEARCH_PATH),
        "view_summary": view_summary,
        "candidate_count_planned": 6,
        "candidate_count_completed": direct_runtime["candidate_count"],
        "training_runtime_seconds": runtime_seconds,
        "full_stage2_scoring": True,
        "stage2_scoring_parallel_jobs": 1,
        "max_stage2_scoring_rows_per_scope": None,
        "random_state": RANDOM_STATE,
    }
    STAGE2_CONFIG_PATH.write_text(json.dumps(config, indent=2, default=_json_default), encoding="utf-8")
    return config


def main():
    start = time.perf_counter()
    raw_path = DATA_DIR / "raw" / "accepted_2007_to_2018Q4.csv"
    print(f"Loading raw loss data from {raw_path}", flush=True)
    raw_loss_df = ensure_sample_id(load_loss_raw_dataset(raw_path))
    workflow_df = prepare_loss_workflow_dataset(raw_loss_df)
    charged = build_charged_off_loss_proxy(workflow_df)
    lgd_lookup = fit_expected_lgd_lookup(charged)
    calibration_reference = _build_workflow_12m_calibration_reference(workflow_df, lgd_lookup)

    print("Fitting static lifetime HGB rerun...", flush=True)
    static_bundle, static_summary = _fit_static_lifetime_model(raw_loss_df)

    print("Preparing direct active-snapshot 12M sample...", flush=True)
    modeling_sample = read_modeling_sample(PROCESSED_DIR / "calendar_survival_modeling_sample.parquet")
    modeling_sample = _enrich_panel_with_workflow_fields(modeling_sample, workflow_df)
    direct_frame = _prepare_direct_frame(modeling_sample, lgd_lookup)
    view, view_summary = _build_direct_training_view(direct_frame)
    view.to_parquet(DIRECT_SAMPLE_PATH, index=False)
    _write_json(DIRECT_SUMMARY_PATH, view_summary)

    print("Training direct active-snapshot 12M candidates...", flush=True)
    direct_bundle, search, direct_runtime = _fit_direct_model(view, calibration_reference)
    search.to_csv(DIRECT_SEARCH_PATH, index=False)

    print("Scoring direct active-snapshot backtests and stage2 predictions...", flush=True)
    test_summary = _write_test_exports(direct_bundle, view, static_bundle, raw_loss_df, workflow_df)
    stage2_predictions = _build_stage2_predictions(static_bundle, direct_bundle, raw_loss_df, workflow_df)
    stage2_predictions.to_csv(STAGE2_PREDICTION_PATH, index=False)

    runtime_seconds = float(time.perf_counter() - start)
    config = _write_config(
        static_summary,
        direct_bundle,
        direct_runtime,
        view_summary,
        runtime_seconds,
    )
    runtime_payload = {
        "training_mode": "direct_active_snapshot_12m_static_lifetime_hazard_comparison",
        "runtime_seconds_including_full_stage2_prediction_export": runtime_seconds,
        "training_budget_seconds": TRAINING_BUDGET_SECONDS,
        "static_lifetime": static_summary,
        "direct_12m": direct_runtime,
        "direct_12m_test": test_summary,
        "search_table_path": str(DIRECT_SEARCH_PATH),
        "stage2_prediction_path": str(STAGE2_PREDICTION_PATH),
        "stage2_config_path": str(STAGE2_CONFIG_PATH),
    }
    _write_json(DIRECT_RUNTIME_PATH, runtime_payload)

    print("Running loss reserve workflow with direct 12M stage2 predictions...", flush=True)
    loss_config = build_loss_config(config)
    loss_config["observable_12m_loss_metrics_path"] = str(OBSERVABLE_12M_METRICS_PATH)
    loss_result = run_loss_reserve_workflow(loss_config)
    export_loss_reserve_visuals(
        loss_result,
        output_dir=REPORT_DIR / "figures" / "loss_reserve",
        show=False,
    )
    portfolio = pd.read_csv(PROCESSED_DIR / "portfolio_expected_loss_summary.csv")
    print(
        json.dumps(
            {
                "selected_direct_12m_model": direct_runtime["selected_model_name"],
                "direct_12m_validation_auc": direct_runtime["selected_validation_auc"],
                "direct_12m_test_auc": test_summary["test_metrics"]["AUC"],
                "runtime_seconds": runtime_seconds,
                "portfolio_summary": portfolio.to_dict(orient="records"),
            },
            indent=2,
            default=_json_default,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
