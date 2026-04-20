from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype

from loss_preprocess import LOSS_CLEANING_METHODS, MANAGERIAL_SEGMENT_COLUMNS


def write_loss_reserve_report(workflow_result, report_config, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    markdown = build_loss_reserve_report(workflow_result, report_config)
    output_path.write_text(markdown, encoding="utf-8")
    return output_path


def build_loss_reserve_report(workflow_result, report_config):
    lines = [
        "# Credit Loss, CECL Proxy, And Risk Segmentation Report",
        "",
        "## Course Formula Alignment",
        "",
        "This phase extends the stage-2 PD workflow into a course-aligned expected-loss framework.",
        "The core lens is `Expected Loss = PD x LGD x EAD`, with installment-loan simplifications for EAD, credibility-weighted LGD lookups, a pooled-logistic hazard model for lifetime PD, and management-facing segmentation.",
        "Pricing and economic capital are not implemented here; the report ends with a short bridge showing how these outputs feed those later modules.",
        "",
        "## Data Assets And Temporal Policy",
        "",
        f"- Stage-2 PD benchmark path: `{report_config['pd_prediction_path']}`",
        f"- Loss workflow dataset path: `{report_config['loss_workflow_path']}`",
        f"- Charged-off LGD/EAD proxy path: `{report_config['charged_off_proxy_path']}`",
        f"- Active reserve snapshot path: `{report_config['cecl_snapshot_path']}`",
        f"- Holdout loss evaluation path: `{report_config['test_loss_metrics_path']}`",
        f"- Segment summary path: `{report_config['segment_summary_path']}`",
        "",
        "The temporal split remains origination-based throughout the workflow: train vintages before 2016, validation vintages in 2016, and test vintages from 2017 onward.",
        "",
    ]

    loss_dataset_summary = workflow_result.get("loss_dataset_summary", pd.DataFrame())
    if not loss_dataset_summary.empty:
        lines.extend(
            [
                "### Workflow Dataset Summary",
                "",
                _dataframe_to_markdown(loss_dataset_summary),
                "",
            ]
        )

    lines.extend(["## Cleaning And Proxy Construction", ""])
    lines.extend([f"- {step}" for step in LOSS_CLEANING_METHODS])
    lines.append("")

    charged_summary = workflow_result.get("charged_off_proxy_summary", {})
    lines.extend(
        [
            "### LGD And EAD Proxy Design",
            "",
            "- `EAD_proxy = max(funded_amnt - total_rec_prncp, 0)` for charged-off loans.",
            "- `net_recoveries = max(recoveries - collection_recovery_fee, 0)`.",
            "- `LGD_proxy = clip(1 - net_recoveries / EAD_proxy, 0, 1)` when `EAD_proxy > 0`.",
            "- `EAD_current` uses `out_prncp` when available and falls back to a scheduled-balance amortization proxy.",
            "",
            f"- Charged-off proxy rows kept: `{charged_summary.get('rows', 'NA')}`",
            f"- Charged-off proxy portfolio LGD mean: `{_format_float(charged_summary.get('portfolio_lgd_mean'))}`",
            f"- Charged-off proxy average EAD: `{_format_float(charged_summary.get('avg_ead_proxy'))}`",
            "",
        ]
    )

    lgd_summary = workflow_result.get("lgd_lookup_summary", {})
    lines.extend(
        [
            "### Expected LGD Lookup",
            "",
            f"- Fit splits: `{', '.join(lgd_summary.get('fit_splits', []))}`",
            f"- Shrinkage rule: `weight = n / (n + {lgd_summary.get('shrinkage_k', 'NA')})`",
            f"- Portfolio fallback LGD: `{_format_float(lgd_summary.get('portfolio_mean'))}`",
            f"- Charged-off training rows: `{lgd_summary.get('training_rows', 'NA')}`",
            f"- Holdout charged-off LGD MAE: `{_format_float(lgd_summary.get('test_lgd_mae'))}`",
            f"- Holdout charged-off LGD RMSE: `{_format_float(lgd_summary.get('test_lgd_rmse'))}`",
            "",
        ]
    )

    lgd_source_table = workflow_result.get("lgd_lookup_source_table", pd.DataFrame())
    if not lgd_source_table.empty:
        lines.extend(
            [
                "Lookup-source usage on the holdout and reserve outputs:",
                "",
                _dataframe_to_markdown(lgd_source_table),
                "",
            ]
        )

    lines.extend(["## Hazard-Based Lifetime PD", ""])
    hazard_panel_summary = workflow_result.get("hazard_panel_summary", pd.DataFrame())
    if not hazard_panel_summary.empty:
        lines.extend(
            [
                "### Hazard Panel Summary",
                "",
                _dataframe_to_markdown(hazard_panel_summary),
                "",
            ]
        )

    hazard_search_table = workflow_result.get("hazard_search_table", pd.DataFrame())
    if not hazard_search_table.empty:
        display_table = hazard_search_table.copy()
        for column in [
            "validation_lifetime_auc",
            "validation_lifetime_ks",
            "validation_lifetime_brier",
            "validation_12m_auc",
            "validation_12m_brier",
        ]:
            if column in display_table.columns:
                display_table[column] = display_table[column].map(_format_float)
        lines.extend(
            [
                "### Hazard Parameter Search",
                "",
                _dataframe_to_markdown(display_table),
                "",
            ]
        )

    for title, key in [
        ("Validation Hazard Metrics", "hazard_validation_overall"),
        ("Test Hazard Metrics", "hazard_test_overall"),
        ("Validation Calibration By Vintage", "hazard_validation_vintage"),
        ("Test Calibration By Vintage", "hazard_test_vintage"),
        ]:
        table = workflow_result.get(key, pd.DataFrame())
        if table.empty:
            continue
        display_table = _format_numeric_columns(
            table.copy(),
            include_tokens=("auc", "ks", "brier", "rate", "pd"),
        )
        lines.extend([f"### {title}", "", _dataframe_to_markdown(display_table), ""])

    lines.extend(["## Expected Loss Outputs", ""])
    portfolio_summary = workflow_result.get("portfolio_summary", pd.DataFrame())
    if not portfolio_summary.empty:
        display_table = portfolio_summary.copy()
        for column in [
            "funded_amount",
            "avg_pd",
            "avg_lgd",
            "avg_ead",
            "avg_el",
            "total_el",
            "portfolio_el_share",
            "actual_loss_rate",
        ]:
            if column in display_table.columns:
                display_table[column] = display_table[column].map(_format_float)
        lines.extend([_dataframe_to_markdown(display_table), ""])

    stage2_summary = workflow_result.get("stage2_benchmark_summary", {})
    if stage2_summary:
        lines.extend(
            [
                "### Fixed-Horizon PD Input For 12-Month EL",
                "",
                f"- Locked stage-2 model: `{stage2_summary.get('model_name', 'NA')}`",
                f"- Prediction source: `{stage2_summary.get('prediction_source', 'NA')}`",
                f"- Rows scored with the fixed-horizon model: `{stage2_summary.get('rows', 'NA')}`",
                f"- Mean fixed-horizon PD across scored rows: `{_format_float(stage2_summary.get('mean_predicted_pd'))}`",
                f"- Used as the 12-month EL PD input: `{stage2_summary.get('used_for_12m_el', False)}`",
                "",
            ]
        )

    resolved_sample = workflow_result.get("resolved_holdout_sample", pd.DataFrame())
    if not resolved_sample.empty:
        display_table = resolved_sample.copy()
        for column in [
            "stage2_champion_pd",
            "pd_12m_fixed_horizon",
            "pd_12m_hazard",
            "lifetime_pd_hazard",
            "expected_lgd",
            "ead_reference",
            "expected_loss_12m",
            "expected_loss_lifetime",
            "actual_net_loss",
        ]:
            if column in display_table.columns:
                display_table[column] = display_table[column].map(_format_float)
        lines.extend(
            [
                "### Holdout Loan-Level Examples",
                "",
                _dataframe_to_markdown(display_table),
                "",
            ]
        )

    active_sample = workflow_result.get("active_snapshot_sample", pd.DataFrame())
    if not active_sample.empty:
        display_table = active_sample.copy()
        for column in [
            "stage2_champion_pd",
            "pd_12m_fixed_horizon",
            "pd_12m_hazard",
            "lifetime_pd_hazard",
            "expected_lgd",
            "ead_current",
            "expected_loss_12m",
            "lifetime_expected_loss",
        ]:
            if column in display_table.columns:
                display_table[column] = display_table[column].map(_format_float)
        lines.extend(
            [
                "## CECL Proxy And Reserve Snapshot",
                "",
                "The reserve view combines lifetime PD from the hazard model with credibility-weighted expected LGD and current EAD.",
                "",
                "### Active Snapshot Examples",
                "",
                _dataframe_to_markdown(display_table),
                "",
            ]
        )

    segment_summary = workflow_result.get("segment_summary", pd.DataFrame())
    if not segment_summary.empty:
        lines.extend(
            [
                "## Risk Segmentation Analysis",
                "",
                f"The management-facing cuts in this phase are `{', '.join(MANAGERIAL_SEGMENT_COLUMNS)}`.",
                "",
            ]
        )
        for analysis_scope in segment_summary["analysis_scope"].dropna().unique():
            scoped = segment_summary[segment_summary["analysis_scope"] == analysis_scope].copy()
            scoped = scoped.sort_values(["pd_measure", "segment_type", "total_el"], ascending=[True, True, False])
            scoped = scoped.groupby(["pd_measure", "segment_type"], as_index=False).head(5)
            display_table = scoped.copy()
            for column in [
                "funded_amount",
                "avg_pd",
                "avg_lgd",
                "avg_ead",
                "avg_el",
                "total_el",
                "portfolio_el_share",
            ]:
                if column in display_table.columns:
                    display_table[column] = display_table[column].map(_format_float)
            lines.extend(
                [
                    f"### Top Segment Contributions: {analysis_scope}",
                    "",
                    _dataframe_to_markdown(display_table),
                    "",
                ]
            )

    lines.extend(
        [
            "## Next-Phase Bridge",
            "",
            "These outputs are now sufficient to support the next course topics without implementing them yet.",
            "- Risk-based pricing can consume the expected-loss estimates as the loss-premium input.",
            "- Economic capital can use the expected-loss baseline together with a later unexpected-loss module.",
            "- Segment-level EL concentration provides the management view needed before capital or pricing overlays are added.",
            "",
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


def _format_numeric_columns(df, include_tokens=(), exact_columns=()):
    exact_columns = set(exact_columns)
    lowered_tokens = tuple(token.lower() for token in include_tokens)

    for column in df.columns:
        column_lower = str(column).lower()
        should_format = column in exact_columns or any(
            token in column_lower for token in lowered_tokens
        )
        if should_format and is_numeric_dtype(df[column]):
            df[column] = df[column].map(_format_float)

    return df
