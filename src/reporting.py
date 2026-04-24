import json
from pathlib import Path

import pandas as pd

from preprocess import CLEANING_METHODS, FEATURE_FLAG_CANDIDATES, FEATURE_GROUPS


def write_temporal_report(
    report_result,
    report_config,
    output_path,
    feature_search_result=None,
    visual_paths=None,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    markdown = build_temporal_report(
        report_result,
        report_config,
        feature_search_result=feature_search_result,
        visual_paths=visual_paths,
    )
    output_path.write_text(markdown, encoding="utf-8")
    return output_path


def build_temporal_report(
    report_result,
    report_config,
    feature_search_result=None,
    visual_paths=None,
):
    visual_paths = visual_paths or {}
    prepared_data = report_result["prepared_data"]
    split_summary = report_result["prepared_data"]["split_summary"].copy()
    validation_table = report_result["validation_table"].copy()
    final_test_table = report_result["final_test_table"].copy()
    champion_period_table = report_result["champion_period_table"].copy()
    importance_tables = report_result["importance_tables"]
    grid_search_table = report_result.get("grid_search_table", pd.DataFrame()).copy()

    for rate_column in ["default_rate", "event_rate_1m"]:
        if rate_column in split_summary.columns:
            split_summary[rate_column] = split_summary[rate_column].map(_format_float)
    for date_column in [
        "min_issue_date",
        "max_issue_date",
        "min_snapshot_month",
        "max_snapshot_month",
    ]:
        if date_column in split_summary.columns:
            split_summary[date_column] = pd.to_datetime(
                split_summary[date_column]
            ).dt.strftime("%Y-%m-%d")

    validation_display = validation_table[
        _existing_columns(
            validation_table,
            [
                "validation_rank",
                "model_name",
                "feature_profile",
                "feature_count",
                "search_mode",
                "search_candidates",
                "params",
                "validation_auc",
                "validation_ks",
                "validation_brier",
                "validation_12m_auc",
                "validation_12m_brier",
                "calibration_summary",
            ],
        )
    ].copy()
    for column in [
        "validation_auc",
        "validation_ks",
        "validation_brier",
        "validation_12m_auc",
        "validation_12m_brier",
    ]:
        if column not in validation_display.columns:
            continue
        validation_display[column] = validation_display[column].map(_format_float)

    final_test_display = final_test_table[
        _existing_columns(
            final_test_table,
            [
                "validation_rank",
                "model_name",
                "feature_profile",
                "feature_count",
                "params",
                "test_auc",
                "test_ks",
                "test_brier",
                "test_12m_auc",
                "test_12m_brier",
                "calibration_summary",
            ],
        )
    ].copy()
    for column in ["test_auc", "test_ks", "test_brier", "test_12m_auc", "test_12m_brier"]:
        if column not in final_test_display.columns:
            continue
        final_test_display[column] = final_test_display[column].map(_format_float)

    champion_period_display = champion_period_table[
        _existing_columns(
            champion_period_table,
            [
                "period",
                "fitted_on",
                "rows",
                "default_rate",
                "event_rate_1m",
                "min_issue_date",
                "max_issue_date",
                "min_snapshot_month",
                "max_snapshot_month",
                "auc",
                "ks",
                "brier",
            ],
        )
    ].copy()
    for column in ["default_rate", "event_rate_1m", "auc", "ks", "brier"]:
        if column in champion_period_display.columns:
            champion_period_display[column] = champion_period_display[column].map(_format_float)
    for date_column in [
        "min_issue_date",
        "max_issue_date",
        "min_snapshot_month",
        "max_snapshot_month",
    ]:
        if date_column in champion_period_display.columns:
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
        grid_search_candidate_count = len(grid_search_display)
        grid_search_display = (
            grid_search_display.groupby("model_name", group_keys=False)
            .head(10)
            .reset_index(drop=True)
        )
        for column in ["validation_auc", "validation_ks", "validation_brier"]:
            grid_search_display[column] = grid_search_display[column].map(_format_float)
    else:
        grid_search_display = pd.DataFrame()
        grid_search_candidate_count = 0

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
        f"- Modeling target: `{prepared_data.get('modeling_type', 'static')}`",
        "",
        _dataframe_to_markdown(split_summary),
        "",
        "## Cleaning Method",
        "",
    ]
    time_estimate = report_config.get("calendar_training_time_estimate_result")
    if time_estimate:
        lines.extend(
            [
                "## Big Data Execution Note",
                "",
                f"- Modeling sample rows used for the estimate: `{time_estimate.get('sample_rows')}`; sampled split counts: `{time_estimate.get('split_counts', {})}`.",
                "- Full calendar exposure counts are stored separately by Dask; the full panel parquet is not written.",
                f"- Full 62-candidate grid estimated runtime: `{_format_float(time_estimate.get('estimated_total_grid_hours'))}` hours.",
                f"- Runtime threshold: `{_format_float(time_estimate.get('threshold_hours'))}` hours.",
                f"- Exceeded threshold: `{bool(time_estimate.get('exceeds_threshold'))}`.",
                f"- Current report uses resource-bounded fixed-parameter training: `{bool(report_config.get('resource_bounded_training', False))}`.",
                f"- Resource-bounded model parameters adjusted for runtime: `{bool(report_config.get('resource_bounded_model_params_adjusted', False))}`.",
                "The resource-bounded run is used to refresh diagnostics and paper text without presenting it as the completed 62-candidate production grid.",
                "",
            ]
        )
    if prepared_data.get("modeling_type") == "calendar_time_hazard":
        cleaning_methods = [
            "Keep resolved and active loan states so the calendar-time survival panel can represent events and censored exposure.",
            "Parse issue_d and last_pymnt_d into monthly dates; performance fields are used only for event/censoring and evaluation.",
            "Build one row per sample_id and snapshot_month with target_1m equal to next-month charge-off/default.",
            "Split rows by snapshot_month; the same loan may legitimately appear across train, validation, and test months.",
            "Convert interest-rate and revolving-utilization percent strings into numeric values.",
            "Convert invalid DTI, int_rate, and revol_util values to missing before imputation.",
            "Impute modeled numeric columns using medians fit on train snapshot rows, then apply the same values to validation, test, and scoring snapshots.",
        ]
    elif prepared_data.get("modeling_type") == "hazard":
        cleaning_methods = [
            "Keep resolved and active loan states (`Fully Paid`, `Charged Off`, `Current`, `In Grace Period`, `Late`, and `Default`) so the hazard panel can represent both events and censored exposure.",
            "Parse issue_d into issue_date and keep it strictly as temporal metadata.",
            "Parse last_pymnt_d into observed months, then cap months_on_book at term_months.",
            "Convert interest-rate and revolving-utilization percent strings into numeric values.",
            "Convert invalid DTI, int_rate, and revol_util values to missing before imputation.",
            "Impute modeled numeric columns using medians fit on the training split, then apply the same values to validation and test.",
            "Build grouped loan-month hazard states and one-hot encode the active hazard profile with drop_first=True.",
        ]
    else:
        cleaning_methods = CLEANING_METHODS
    lines.extend([f"- {step}" for step in cleaning_methods])

    if prepared_data.get("modeling_type") in {"hazard", "calendar_time_hazard"}:
        champion_key = report_result.get("selected_model_key")
        champion_model = report_result.get("final_models", {}).get(champion_key, {})
        champion_profile = champion_model.get("feature_profile", "hazard_profile")
        champion_feature_columns = champion_model.get("feature_columns", feature_columns)
        if prepared_data.get("modeling_type") == "calendar_time_hazard":
            feature_lines = [
                "- Calendar-time inputs include `snapshot_month`, `calendar_year`, `calendar_quarter`, `month_on_book`, `remaining_term`, `seasoning_ratio`, vintage fields, and scheduled-balance proxies.",
                "- Full calendar hazard profile is used for RF/XGBoost/MLP; Logistic uses a compact profile without purpose dummies or high-order interaction columns.",
                "- Enabled main model families are Logistic Regression, Random Forest, XGBoost, and MLP Neural Network; HistGradientBoosting is excluded from the main PD search.",
            ]
        else:
            feature_lines = [
                "- Hazard state inputs: `grade`, `fico_bucket`, `purpose_group`, `annual_income_band`, `term_months`, and `month_bucket`.",
                "- Full grouped hazard profile for RF/XGB adds `term_month_interaction`.",
                "- Compact grouped hazard profile for Logistic removes the higher-dimensional interaction block.",
            ]
        lines.extend(
            [
                "",
                "## Selected Hazard Features",
                "",
                *feature_lines,
                f"- Champion feature profile: `{champion_profile}`.",
                f"- Champion modeled column count: `{len(champion_feature_columns)}`.",
                "- In the dual-PD final workflow, `predicted_pd` is the rerun static HGB lifetime PD; `predicted_pd_12m` is the fixed-horizon calendar-time hazard PD.",
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
    else:
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
    validation_visual_lines = _image_section_lines(
        [
            ("Observed Default Rate Trend", visual_paths.get("default_rate_trend")),
            ("Validation ROC Comparison", visual_paths.get("validation_roc_comparison")),
            (
                "Validation Reliability Diagram",
                visual_paths.get("validation_calibration_comparison"),
            ),
        ]
    )
    if validation_visual_lines:
        lines.extend(
            [
                "## Validation Visual Diagnostics",
                "",
                "Calibration markers are sized by bin count so sparse tail bins are visually distinguishable from dense central bins.",
                "",
            ]
            + validation_visual_lines
        )

    if not grid_search_display.empty:
        lines.extend(
            [
                "## Grid Search Results",
                "",
                f"The table below shows the top 10 validation-ranked candidates per model family out of `{grid_search_candidate_count}` total grid-search candidates.",
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
        feature_visual_lines = _image_section_lines(
            [("Feature Search Impact", visual_paths.get("feature_search_deltas"))]
        )
        if feature_visual_lines:
            lines.extend(
                [
                    "## Feature Discovery Visuals",
                    "",
                    "The feature-search chart annotates exact validation deltas and shades the near-zero zone to separate material changes from noise-level movement.",
                    "",
                ]
                + feature_visual_lines
            )

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
        ]
    )

    holdout_visual_lines = _image_section_lines(
        [
            ("Test ROC Comparison", visual_paths.get("test_roc_comparison")),
            ("Test Reliability Diagram", visual_paths.get("test_calibration_comparison")),
            ("Champion Validation KS Curve", visual_paths.get("champion_validation_ks")),
            ("Champion Test KS Curve", visual_paths.get("champion_test_ks")),
            ("Champion Score Distribution", visual_paths.get("champion_score_distribution")),
        ]
    )
    if holdout_visual_lines:
        lines.extend(["## Holdout Visual Diagnostics", ""] + holdout_visual_lines)

    lines.extend(
        [
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

    interpretability_visual_lines = _image_section_lines(
        [
            ("Logistic Coefficients", visual_paths.get("logistic_coefficients")),
            ("XGBoost Feature Importance", visual_paths.get("xgboost_feature_importance")),
        ]
    )
    if interpretability_visual_lines:
        lines.extend(["## Visual Interpretability", ""] + interpretability_visual_lines)

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


def _existing_columns(df, columns):
    return [column for column in columns if column in df.columns]


def _image_section_lines(image_specs):
    lines = []
    for title, path in image_specs:
        if not path:
            continue
        lines.extend([f"### {title}", "", _image_markdown(title, path), ""])
    return lines


def _image_markdown(title, path):
    return f"![{title}]({Path(path).resolve()})"
