import json
from pathlib import Path

import pandas as pd

from preprocess import CLEANING_METHODS, FEATURE_FLAG_CANDIDATES, FEATURE_GROUPS


def write_temporal_report(report_result, report_config, output_path, feature_search_result=None):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    markdown = build_temporal_report(
        report_result,
        report_config,
        feature_search_result=feature_search_result,
    )
    output_path.write_text(markdown, encoding="utf-8")
    return output_path


def build_temporal_report(report_result, report_config, feature_search_result=None):
    prepared_data = report_result["prepared_data"]
    split_summary = report_result["prepared_data"]["split_summary"].copy()
    validation_table = report_result["validation_table"].copy()
    final_test_table = report_result["final_test_table"].copy()
    champion_period_table = report_result["champion_period_table"].copy()
    importance_tables = report_result["importance_tables"]
    grid_search_table = report_result.get("grid_search_table", pd.DataFrame()).copy()

    split_summary["default_rate"] = split_summary["default_rate"].map(_format_float)
    for date_column in ["min_issue_date", "max_issue_date"]:
        split_summary[date_column] = pd.to_datetime(split_summary[date_column]).dt.strftime(
            "%Y-%m-%d"
        )

    validation_display = validation_table[
        [
            "validation_rank",
            "model_name",
            "search_mode",
            "search_candidates",
            "params",
            "validation_auc",
            "validation_ks",
            "validation_brier",
            "calibration_summary",
        ]
    ].copy()
    for column in ["validation_auc", "validation_ks", "validation_brier"]:
        validation_display[column] = validation_display[column].map(_format_float)

    final_test_display = final_test_table[
        [
            "validation_rank",
            "model_name",
            "params",
            "test_auc",
            "test_ks",
            "test_brier",
            "calibration_summary",
        ]
    ].copy()
    for column in ["test_auc", "test_ks", "test_brier"]:
        final_test_display[column] = final_test_display[column].map(_format_float)

    champion_period_display = champion_period_table[
        [
            "period",
            "fitted_on",
            "rows",
            "default_rate",
            "min_issue_date",
            "max_issue_date",
            "auc",
            "ks",
            "brier",
        ]
    ].copy()
    for column in ["default_rate", "auc", "ks", "brier"]:
        champion_period_display[column] = champion_period_display[column].map(_format_float)
    for date_column in ["min_issue_date", "max_issue_date"]:
        champion_period_display[date_column] = pd.to_datetime(
            champion_period_display[date_column]
        ).dt.strftime("%Y-%m-%d")

    if not grid_search_table.empty:
        grid_search_display = grid_search_table[
            [
                "model_name",
                "search_rank",
                "params",
                "validation_auc",
                "validation_ks",
                "validation_brier",
            ]
        ].copy()
        for column in ["validation_auc", "validation_ks", "validation_brier"]:
            grid_search_display[column] = grid_search_display[column].map(_format_float)
    else:
        grid_search_display = pd.DataFrame()

    champion_validation = report_result["champion_validation"]
    champion_test = report_result["champion_test"]
    feature_columns = prepared_data["feature_columns"]
    active_feature_flags = {
        key: value
        for key, value in report_config.get("feature_flags", {}).items()
        if value
    }

    lines = [
        "# Temporal Credit Risk Modeling Report",
        "",
        "## Objective And Leakage Rule",
        "",
        "This report documents a chronologically valid probability-of-default workflow.",
        "The project uses issue_date only as temporal metadata and never as a predictive feature.",
        "Grid search and model selection are restricted to the training and validation periods.",
        "The final holdout test period is opened only after the model family, features, and hyperparameters are locked.",
        "",
        "## Dataset And Temporal Split",
        "",
        f"- Processed dataset path: `{report_config['processed_data_path']}`",
        f"- Prediction export path: `{report_config['output_path']}`",
        f"- Report sample fraction: `{report_config.get('sample_frac', 1.0)}`",
        f"- Grid search enabled: `{bool(report_config.get('use_grid_search', False))}`",
        "",
        _dataframe_to_markdown(split_summary),
        "",
        "## Cleaning Method",
        "",
    ]
    lines.extend([f"- {step}" for step in CLEANING_METHODS])

    lines.extend(
        [
            "",
            "## Selected Features",
            "",
            f"- Core numeric features: `{', '.join(FEATURE_GROUPS['core_numeric_features'])}`",
            f"- Additional numeric features: `{', '.join(FEATURE_GROUPS['additional_numeric_features'])}`",
            f"- Base engineered features: `{', '.join(FEATURE_GROUPS['base_engineered_features'])}`",
            f"- Extended engineered features: `{', '.join(FEATURE_GROUPS['extended_engineered_features'])}`",
            f"- Dummy groups available: `{', '.join(FEATURE_GROUPS['categorical_dummy_groups'])}`",
            f"- Active feature flags: `{', '.join(sorted(active_feature_flags))}`",
            f"- Final modeled column count: `{len(feature_columns)}`",
            "",
            "### Encoded Feature Columns",
            "",
            ", ".join(feature_columns),
            "",
            "## Candidate Models And Search Space",
            "",
            f"- Enabled model families: `{', '.join(report_config['enabled_models'])}`",
            f"- Grid search models: `{', '.join(report_config.get('grid_search_models', []))}`",
            "",
        ]
    )

    for model_name in report_config["enabled_models"]:
        lines.extend(
            [
                f"### {model_name}",
                "",
                "Base parameters:",
                "",
                "```json",
                _pretty_json(report_config["model_params"].get(model_name, {})),
                "```",
                "",
                "Grid search space:",
                "",
                "```json",
                _pretty_json(report_config.get("param_grid", {}).get(model_name, {})),
                "```",
                "",
            ]
        )

    lines.extend(
        [
            "## Validation Ranking",
            "",
            _dataframe_to_markdown(validation_display),
            "",
        ]
    )

    if not grid_search_display.empty:
        lines.extend(
            [
                "## Grid Search Results",
                "",
                _dataframe_to_markdown(grid_search_display),
                "",
            ]
        )

    if feature_search_result is not None:
        feature_search_table = feature_search_result.get("feature_search_table", pd.DataFrame()).copy()
        baseline_metrics = feature_search_result.get("baseline_metrics", {})
        if not feature_search_table.empty:
            feature_search_display = feature_search_table[
                [
                    "feature_rank",
                    "feature_group",
                    "feature_flag",
                    "search_model",
                    "search_mode",
                    "params",
                    "added_feature_count",
                    "added_columns_preview",
                    "validation_auc",
                    "validation_ks",
                    "validation_brier",
                    "delta_auc",
                    "delta_ks",
                    "delta_brier",
                ]
            ].copy()
            for column in [
                "validation_auc",
                "validation_ks",
                "validation_brier",
                "delta_auc",
                "delta_ks",
                "delta_brier",
            ]:
                feature_search_display[column] = feature_search_display[column].map(_format_float)
        else:
            feature_search_display = pd.DataFrame()

        lines.extend(
            [
                "## Feature Discovery On Omitted Feature Groups",
                "",
                f"- Feature-search model: `{feature_search_result.get('search_model_name', 'NA')}`",
                f"- Baseline validation AUC: `{_format_float(baseline_metrics.get('validation_auc'))}`",
                f"- Baseline validation KS: `{_format_float(baseline_metrics.get('validation_ks'))}`",
                f"- Baseline validation Brier: `{_format_float(baseline_metrics.get('validation_brier'))}`",
                "",
            ]
        )
        if not feature_search_display.empty:
            lines.extend([_dataframe_to_markdown(feature_search_display), ""])
        else:
            lines.extend(["No omitted feature groups were screened.", ""])

    lines.extend(
        [
            "## Final Chosen Hyperparameters",
            "",
            f"- Selected model: `{champion_validation['model_name']}`",
            f"- Validation AUC: `{_format_float(champion_validation['validation_auc'])}`",
            f"- Validation KS: `{_format_float(champion_validation['validation_ks'])}`",
            f"- Validation Brier: `{_format_float(champion_validation['validation_brier'])}`",
            "",
            "```json",
            champion_validation["params"],
            "```",
            "",
            "## Holdout Test Results",
            "",
            f"- Champion test AUC: `{_format_float(champion_test['test_auc'])}`",
            f"- Champion test KS: `{_format_float(champion_test['test_ks'])}`",
            f"- Champion test Brier: `{_format_float(champion_test['test_brier'])}`",
            "",
            _dataframe_to_markdown(final_test_display),
            "",
            "## Champion Temporal Performance",
            "",
            _dataframe_to_markdown(champion_period_display),
            "",
            "## Feature Importance And Interpretability",
            "",
            "### Logistic Regression Coefficients",
            "",
        ]
    )

    logistic_table = importance_tables.get("logistic_regression")
    if logistic_table is not None and not logistic_table.empty:
        display_table = logistic_table.copy()
        for column in ["coefficient", "abs_coefficient"]:
            display_table[column] = display_table[column].map(_format_float)
        lines.extend([_dataframe_to_markdown(display_table), ""])
    else:
        lines.extend(["Logistic regression coefficients were not available.", ""])

    lines.extend(["### XGBoost Feature Importance", ""])
    xgboost_table = importance_tables.get("xgboost")
    if xgboost_table is not None and not xgboost_table.empty:
        display_table = xgboost_table.copy()
        display_table["importance"] = display_table["importance"].map(_format_float)
        lines.extend([_dataframe_to_markdown(display_table), ""])
    else:
        lines.extend(["XGBoost feature importance was not available.", ""])

    lines.extend(
        [
            "## Candidate Feature Flags",
            "",
        ]
    )
    lines.extend(
        [
            f"- `{flag_name}`: {description}"
            for flag_name, description in FEATURE_FLAG_CANDIDATES.items()
        ]
    )

    return "\n".join(lines).strip() + "\n"


def _dataframe_to_markdown(df):
    display_df = df.fillna("")
    columns = list(display_df.columns)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(row[column]) for column in columns) + " |"
        for _, row in display_df.iterrows()
    ]
    return "\n".join([header, divider] + rows)


def _format_float(value):
    if pd.isna(value):
        return "NA"
    return f"{float(value):.4f}"


def _pretty_json(payload):
    return json.dumps(payload, indent=2)
