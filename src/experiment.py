import copy
import json

import pandas as pd
from sklearn.model_selection import ParameterGrid

from evaluation import evaluate_model
from model_HGB import build_hist_gradient_boosting_model
from model_LR import build_logistic_model
from model_RF import build_random_forest_model
from model_XGB import build_xgboost_model
from preprocess import (
    FEATURE_FLAG_CANDIDATES,
    ID_COLUMN,
    ISSUE_DATE_COLUMN,
    TARGET_COLUMN,
    ensure_model_ready_dataset,
    select_feature_columns,
)
from split import (
    DEFAULT_TEMPORAL_SPLIT,
    combine_training_frames,
    temporal_train_valid_test_split,
)


MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "hist_gradient_boosting": "HistGradientBoosting",
    "xgboost": "XGBoost",
}
MODEL_BUILDERS = {
    "logistic_regression": build_logistic_model,
    "random_forest": build_random_forest_model,
    "hist_gradient_boosting": build_hist_gradient_boosting_model,
    "xgboost": build_xgboost_model,
}


def prepare_experiment_data(config, df):
    working = ensure_model_ready_dataset(df)
    temporal_split = copy.deepcopy(DEFAULT_TEMPORAL_SPLIT)
    temporal_split.update(config.get("temporal_split", {}))

    feature_columns = select_feature_columns(
        working,
        config.get("feature_flags"),
    )
    if not feature_columns:
        raise ValueError("No feature columns were selected by the current feature flags.")

    meta_columns = [
        column
        for column in config.get(
            "meta_columns",
            [
                ID_COLUMN,
                ISSUE_DATE_COLUMN,
                "loan_amnt",
                "annual_inc",
                "fico_range_low",
                "term_months",
            ],
        )
        if column in working.columns
    ]

    selected_columns = list(dict.fromkeys(meta_columns + feature_columns + [TARGET_COLUMN]))
    experiment_df = working[selected_columns].copy()

    split_payload = temporal_train_valid_test_split(
        experiment_df,
        feature_columns=feature_columns,
        target_col=TARGET_COLUMN,
        date_col=ISSUE_DATE_COLUMN,
        meta_columns=meta_columns,
        train_end=temporal_split["train_end"],
        valid_end=temporal_split["valid_end"],
        sample_frac=float(config.get("sample_frac", 1.0)),
        random_state=int(config.get("random_state", 42)),
    )

    return {
        "df": experiment_df,
        "feature_columns": feature_columns,
        "meta_columns": meta_columns,
        "train": split_payload["train"],
        "validation": split_payload["validation"],
        "test": split_payload["test"],
        "split_summary": split_payload["split_summary"],
        "temporal_bounds": split_payload["temporal_bounds"],
    }


def run_experiment(config, df):
    prepared = prepare_experiment_data(config, df)
    train_payload = prepared["train"]
    validation_payload = prepared["validation"]
    test_payload = prepared["test"]
    train_validation_payload = combine_training_frames(train_payload, validation_payload)

    validation_rows = []
    validation_predictions = {}
    train_only_models = {}
    best_params_by_model = {}
    grid_search_tables = {}

    for model_name in config.get("enabled_models", []):
        search_result = _search_model(
            model_name=model_name,
            config=config,
            train_payload=train_payload,
            validation_payload=validation_payload,
        )
        best_params = search_result["best_params"]
        best_params_by_model[model_name] = best_params
        grid_search_tables[model_name] = search_result["search_table"]
        train_only_models[model_name] = search_result["best_model"]
        validation_predictions[model_name] = search_result["prediction_frame"]

        best_row = search_result["best_row"]
        validation_rows.append(
            {
                "model_key": model_name,
                "model_name": MODEL_LABELS[model_name],
                "params": json.dumps(best_params, sort_keys=True),
                "feature_count": len(prepared["feature_columns"]),
                "search_mode": search_result["search_mode"],
                "search_candidates": search_result["search_candidates"],
                "validation_auc": best_row["validation_auc"],
                "validation_ks": best_row["validation_ks"],
                "validation_brier": best_row["validation_brier"],
                "calibration_summary": best_row["calibration_summary"],
                "validation_metrics": best_row["validation_metrics"],
            }
        )

    validation_table = rank_validation_results(pd.DataFrame(validation_rows))
    validation_table["validation_rank"] = range(1, len(validation_table) + 1)
    champion_row = validation_table.iloc[0].to_dict()
    champion_key = champion_row["model_key"]

    final_test_rows = []
    final_models = {}
    test_predictions = {}
    for model_name in config.get("enabled_models", []):
        best_params = best_params_by_model[model_name]
        final_model = _build_model(model_name, best_params)
        final_model.fit(train_validation_payload["X"], train_validation_payload["y"])
        final_models[model_name] = final_model

        test_pred = final_model.predict_proba(test_payload["X"])[:, 1]
        test_metrics = evaluate_model(test_payload["y"], test_pred)
        test_predictions[model_name] = _build_prediction_frame(
            test_payload["meta"],
            test_payload["y"],
            test_pred,
            model_name,
        )

        validation_rank = int(
            validation_table.loc[
                validation_table["model_key"] == model_name,
                "validation_rank",
            ].iloc[0]
        )
        final_test_rows.append(
            {
                "model_key": model_name,
                "model_name": MODEL_LABELS[model_name],
                "validation_rank": validation_rank,
                "params": json.dumps(best_params, sort_keys=True),
                "search_mode": "refit_on_train_plus_validation",
                "test_auc": test_metrics["AUC"],
                "test_ks": test_metrics["KS"],
                "test_brier": test_metrics["Brier"],
                "calibration_summary": test_metrics["calibration_summary"],
                "test_metrics": test_metrics,
            }
        )

    final_test_table = (
        pd.DataFrame(final_test_rows)
        .sort_values("validation_rank")
        .reset_index(drop=True)
    )
    champion_test_row = final_test_table.loc[
        final_test_table["model_key"] == champion_key
    ].iloc[0].to_dict()

    champion_period_table = _build_champion_period_table(
        champion_key,
        train_payload,
        validation_payload,
        test_payload,
        train_only_models[champion_key],
        final_models[champion_key],
    )

    importance_tables = _build_importance_tables(
        final_models=final_models,
        feature_columns=prepared["feature_columns"],
    )

    combined_grid_search_table = _combine_search_tables(grid_search_tables)

    return {
        "validation_table": validation_table,
        "final_test_table": final_test_table,
        "champion_validation": champion_row,
        "champion_test": champion_test_row,
        "champion_period_table": champion_period_table,
        "champion_predictions": test_predictions[champion_key],
        "validation_predictions_by_model": validation_predictions,
        "test_predictions_by_model": test_predictions,
        "train_only_models": train_only_models,
        "final_models": final_models,
        "best_params_by_model": best_params_by_model,
        "grid_search_tables": grid_search_tables,
        "grid_search_table": combined_grid_search_table,
        "prepared_data": prepared,
        "importance_tables": importance_tables,
        "selected_model_key": champion_key,
    }


def run_manual_sweep(config, df, model_name):
    prepared = prepare_experiment_data(config, df)
    train_payload = prepared["train"]
    validation_payload = prepared["validation"]
    sweep_candidates = config.get("manual_sweep", {}).get(model_name, [])
    rows = []

    for index, param_override in enumerate(sweep_candidates, start=1):
        candidate_params = copy.deepcopy(config.get("model_params", {}).get(model_name, {}))
        candidate_params.update(param_override)

        model = _build_model(model_name, candidate_params)
        model.fit(train_payload["X"], train_payload["y"])
        validation_pred = model.predict_proba(validation_payload["X"])[:, 1]
        validation_metrics = evaluate_model(validation_payload["y"], validation_pred)
        rows.append(
            {
                "sweep_id": index,
                "model_key": model_name,
                "model_name": MODEL_LABELS[model_name],
                "parameter_override": json.dumps(param_override, sort_keys=True),
                "validation_auc": validation_metrics["AUC"],
                "validation_ks": validation_metrics["KS"],
                "validation_brier": validation_metrics["Brier"],
                "calibration_summary": validation_metrics["calibration_summary"],
            }
        )

    return rank_validation_results(pd.DataFrame(rows))


def run_feature_search(config, df):
    feature_search_config = copy.deepcopy(config.get("feature_search", {}))
    candidate_flags = feature_search_config.get(
        "candidate_flags",
        list(FEATURE_FLAG_CANDIDATES.keys()),
    )
    search_model = feature_search_config.get(
        "search_model",
        config.get("enabled_models", ["xgboost"])[0],
    )

    baseline_config = copy.deepcopy(config)
    baseline_config["enabled_models"] = [search_model]
    for flag in candidate_flags:
        baseline_config.setdefault("feature_flags", {})[flag] = False

    baseline_result = _evaluate_feature_flag_bundle(
        baseline_config,
        df,
        search_model,
    )
    baseline_metrics = baseline_result["best_row"]
    baseline_feature_columns = baseline_result["prepared_data"]["feature_columns"]

    rows = []
    for flag_name in candidate_flags:
        candidate_config = copy.deepcopy(baseline_config)
        candidate_config.setdefault("feature_flags", {})[flag_name] = True
        candidate_result = _evaluate_feature_flag_bundle(
            candidate_config,
            df,
            search_model,
        )

        candidate_metrics = candidate_result["best_row"]
        candidate_feature_columns = candidate_result["prepared_data"]["feature_columns"]
        added_columns = [
            column
            for column in candidate_feature_columns
            if column not in baseline_feature_columns
        ]
        rows.append(
            {
                "feature_flag": flag_name,
                "feature_group": FEATURE_FLAG_CANDIDATES.get(flag_name, flag_name),
                "search_model": MODEL_LABELS[search_model],
                "search_mode": candidate_result["search_mode"],
                "params": json.dumps(candidate_result["best_params"], sort_keys=True),
                "feature_count": len(candidate_feature_columns),
                "added_feature_count": len(added_columns),
                "added_columns_preview": ", ".join(added_columns[:8]),
                "validation_auc": candidate_metrics["validation_auc"],
                "validation_ks": candidate_metrics["validation_ks"],
                "validation_brier": candidate_metrics["validation_brier"],
                "delta_auc": candidate_metrics["validation_auc"] - baseline_metrics["validation_auc"],
                "delta_ks": candidate_metrics["validation_ks"] - baseline_metrics["validation_ks"],
                "delta_brier": candidate_metrics["validation_brier"] - baseline_metrics["validation_brier"],
                "calibration_summary": candidate_metrics["calibration_summary"],
            }
        )

    feature_search_table = rank_feature_search_results(pd.DataFrame(rows))
    if not feature_search_table.empty:
        feature_search_table["feature_rank"] = range(1, len(feature_search_table) + 1)

    return {
        "baseline_metrics": baseline_metrics,
        "baseline_feature_columns": baseline_feature_columns,
        "search_model_key": search_model,
        "search_model_name": MODEL_LABELS[search_model],
        "feature_search_table": feature_search_table,
    }


def fit_locked_model(config, df, model_name=None):
    prepared = prepare_experiment_data(config, df)
    selected_model = model_name or config.get("selected_model_key")
    if selected_model is None:
        enabled_models = config.get("enabled_models", [])
        if len(enabled_models) != 1:
            raise ValueError(
                "fit_locked_model requires selected_model_key or exactly one enabled model."
            )
        selected_model = enabled_models[0]

    params = copy.deepcopy(config.get("model_params", {}).get(selected_model, {}))
    train_validation_payload = combine_training_frames(
        prepared["train"],
        prepared["validation"],
    )
    model = _build_model(selected_model, params)
    model.fit(train_validation_payload["X"], train_validation_payload["y"])

    return {
        "model_key": selected_model,
        "model_name": MODEL_LABELS[selected_model],
        "params": params,
        "model": model,
        "feature_columns": prepared["feature_columns"],
        "meta_columns": prepared["meta_columns"],
        "prepared_data": prepared,
    }


def score_trained_model(model_bundle, feature_df, output_column="predicted_pd"):
    meta_columns = [
        column
        for column in model_bundle.get("meta_columns", [])
        if column in feature_df.columns
    ]
    prediction_frame = feature_df[meta_columns].copy() if meta_columns else pd.DataFrame()
    aligned_features = _align_feature_frame(
        feature_df,
        model_bundle["feature_columns"],
    )
    prediction_frame[output_column] = model_bundle["model"].predict_proba(aligned_features)[:, 1]
    prediction_frame["model_name"] = model_bundle["model_name"]
    return prediction_frame


def rank_validation_results(results_df):
    if results_df.empty:
        return results_df.copy()

    return results_df.sort_values(
        by=["validation_auc", "validation_ks", "validation_brier"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def rank_feature_search_results(results_df):
    if results_df.empty:
        return results_df.copy()

    return results_df.sort_values(
        by=["delta_auc", "delta_ks", "delta_brier", "validation_auc"],
        ascending=[False, False, True, False],
    ).reset_index(drop=True)


def _search_model(model_name, config, train_payload, validation_payload):
    parameter_candidates, search_mode = _parameter_candidates(model_name, config)
    rows = []
    best_key = None
    best_model = None
    best_params = None
    best_metrics = None
    best_prediction_frame = None

    for candidate_id, params in enumerate(parameter_candidates, start=1):
        model = _build_model(model_name, params)
        model.fit(train_payload["X"], train_payload["y"])
        validation_pred = model.predict_proba(validation_payload["X"])[:, 1]
        validation_metrics = evaluate_model(validation_payload["y"], validation_pred)
        comparison_key = _metric_comparison_key(validation_metrics)

        rows.append(
            {
                "candidate_id": candidate_id,
                "model_key": model_name,
                "model_name": MODEL_LABELS[model_name],
                "search_mode": search_mode,
                "params": json.dumps(params, sort_keys=True),
                "params_payload": copy.deepcopy(params),
                "validation_auc": validation_metrics["AUC"],
                "validation_ks": validation_metrics["KS"],
                "validation_brier": validation_metrics["Brier"],
                "calibration_summary": validation_metrics["calibration_summary"],
                "validation_metrics": validation_metrics,
            }
        )

        if best_key is None or comparison_key > best_key:
            best_key = comparison_key
            best_model = model
            best_params = copy.deepcopy(params)
            best_metrics = validation_metrics
            best_prediction_frame = _build_prediction_frame(
                validation_payload["meta"],
                validation_payload["y"],
                validation_pred,
                model_name,
            )

    search_table = rank_validation_results(pd.DataFrame(rows))
    if not search_table.empty:
        search_table["search_rank"] = range(1, len(search_table) + 1)

    best_row = {
        "validation_auc": best_metrics["AUC"],
        "validation_ks": best_metrics["KS"],
        "validation_brier": best_metrics["Brier"],
        "calibration_summary": best_metrics["calibration_summary"],
        "validation_metrics": best_metrics,
    }

    return {
        "best_model": best_model,
        "best_params": best_params,
        "best_row": best_row,
        "prediction_frame": best_prediction_frame,
        "search_table": search_table,
        "search_mode": search_mode,
        "search_candidates": len(parameter_candidates),
    }


def _parameter_candidates(model_name, config):
    base_params = copy.deepcopy(config.get("model_params", {}).get(model_name, {}))
    use_grid_search = bool(config.get("use_grid_search", False))
    grid_search_models = set(
        config.get("grid_search_models", config.get("enabled_models", []))
    )
    if not use_grid_search or model_name not in grid_search_models:
        return [base_params], "fixed"

    grid_spec = copy.deepcopy(config.get("param_grid", {}).get(model_name, {}))
    if not grid_spec:
        return [base_params], "fixed"

    parameter_candidates = []
    for override in ParameterGrid(grid_spec):
        candidate_params = copy.deepcopy(base_params)
        candidate_params.update(override)
        parameter_candidates.append(candidate_params)

    if not parameter_candidates:
        return [base_params], "fixed"
    return parameter_candidates, "grid_search"


def _evaluate_feature_flag_bundle(config, df, model_name):
    prepared = prepare_experiment_data(config, df)
    search_result = _search_model(
        model_name=model_name,
        config=config,
        train_payload=prepared["train"],
        validation_payload=prepared["validation"],
    )
    return {
        "prepared_data": prepared,
        "best_row": search_result["best_row"],
        "best_params": search_result["best_params"],
        "search_mode": search_result["search_mode"],
    }


def _combine_search_tables(grid_search_tables):
    combined_tables = []
    for model_name, table in grid_search_tables.items():
        if table is None or table.empty:
            continue
        combined = table.copy()
        combined["model_key"] = model_name
        combined_tables.append(combined)

    if not combined_tables:
        return pd.DataFrame()

    return pd.concat(combined_tables, ignore_index=True)


def _build_model(model_name, params):
    if model_name not in MODEL_BUILDERS:
        raise KeyError(f"Unsupported model: {model_name}")
    return MODEL_BUILDERS[model_name](params)


def _align_feature_frame(df, feature_columns):
    aligned = df.copy()
    for column in feature_columns:
        if column not in aligned.columns:
            aligned[column] = 0
    return aligned[feature_columns].copy()


def _build_prediction_frame(meta, y_true, y_pred_proba, model_name):
    prediction_frame = meta.copy() if meta is not None else pd.DataFrame()
    prediction_frame["actual_default"] = y_true.to_numpy()
    prediction_frame["predicted_pd"] = y_pred_proba
    prediction_frame["model_name"] = MODEL_LABELS[model_name]

    if ID_COLUMN in prediction_frame.columns:
        prediction_frame = prediction_frame.sort_values(ID_COLUMN).reset_index(drop=True)
    return prediction_frame


def _build_champion_period_table(
    champion_key,
    train_payload,
    validation_payload,
    test_payload,
    train_only_model,
    final_model,
):
    train_pred = train_only_model.predict_proba(train_payload["X"])[:, 1]
    validation_pred = train_only_model.predict_proba(validation_payload["X"])[:, 1]
    test_pred = final_model.predict_proba(test_payload["X"])[:, 1]

    train_metrics = evaluate_model(train_payload["y"], train_pred)
    validation_metrics = evaluate_model(validation_payload["y"], validation_pred)
    test_metrics = evaluate_model(test_payload["y"], test_pred)

    return pd.DataFrame(
        [
            _period_row(
                "train",
                "train only",
                train_payload["df"],
                train_metrics,
            ),
            _period_row(
                "validation",
                "train only",
                validation_payload["df"],
                validation_metrics,
            ),
            _period_row(
                "test",
                "train + validation",
                test_payload["df"],
                test_metrics,
            ),
        ]
    ).assign(model_name=MODEL_LABELS[champion_key])


def _period_row(period_name, fitted_on, partition_df, metrics):
    return {
        "period": period_name,
        "fitted_on": fitted_on,
        "rows": int(len(partition_df)),
        "default_rate": float(partition_df[TARGET_COLUMN].mean()),
        "min_issue_date": partition_df[ISSUE_DATE_COLUMN].min(),
        "max_issue_date": partition_df[ISSUE_DATE_COLUMN].max(),
        "auc": metrics["AUC"],
        "ks": metrics["KS"],
        "brier": metrics["Brier"],
    }


def _build_importance_tables(final_models, feature_columns, top_n=15):
    importance_tables = {}

    logistic_model = final_models.get("logistic_regression")
    if logistic_model is not None:
        coefficients = logistic_model.named_steps["classifier"].coef_[0]
        coefficient_table = pd.DataFrame(
            {
                "feature": feature_columns,
                "coefficient": coefficients,
                "abs_coefficient": abs(coefficients),
            }
        )
        importance_tables["logistic_regression"] = coefficient_table.sort_values(
            "abs_coefficient",
            ascending=False,
        ).head(top_n).reset_index(drop=True)

    xgboost_model = final_models.get("xgboost")
    if xgboost_model is not None:
        importance_table = pd.DataFrame(
            {
                "feature": feature_columns,
                "importance": xgboost_model.feature_importances_,
            }
        )
        importance_tables["xgboost"] = importance_table.sort_values(
            "importance",
            ascending=False,
        ).head(top_n).reset_index(drop=True)

    return importance_tables


def _metric_comparison_key(metrics):
    return (
        float(metrics["AUC"]),
        float(metrics["KS"]),
        -float(metrics["Brier"]),
    )
