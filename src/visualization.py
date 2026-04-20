from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter


MONTH_BUCKET_ORDER = [
    "m01_03",
    "m04_06",
    "m07_12",
    "m13_24",
    "m25_36",
    "m37_48",
    "m49_60",
]

DEFAULT_RANDOM_STATE = 42
SCOPE_ORDER = ["resolved_test", "active_snapshot"]
SCOPE_LABELS = {
    "resolved_test": "Resolved Holdout",
    "active_snapshot": "Active Snapshot",
    "validation": "Validation",
    "test": "Test",
}
SCOPE_COLORS = {
    "resolved_test": "#1f77b4",
    "active_snapshot": "#ff7f0e",
    "validation": "#2ca02c",
    "test": "#d62728",
}
PD_MEASURE_ORDER = ["12m", "lifetime"]
PD_MEASURE_LABELS = {"12m": "12M", "lifetime": "Lifetime"}
SEGMENT_SORT_ORDERS = {
    "grade": ["A", "B", "C", "D", "E", "F", "G"],
    "fico_bucket": ["subprime", "fair", "good", "very_good"],
    "purpose_group": [
        "debt_consolidation",
        "credit_card",
        "home_improvement",
        "major_purchase",
        "other",
    ],
    "annual_income_band": ["<50k", "50-100k", "100-150k", "150k+"],
}


def export_temporal_visuals(
    report_result,
    output_dir,
    feature_search_result=None,
    show=False,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    visual_paths = {}
    prepared_df = report_result["prepared_data"]["df"].copy()
    champion_key = report_result["selected_model_key"]
    champion_name = report_result["champion_test"]["model_name"]

    validation_metric_rows = [
        (row.model_name, row.validation_metrics)
        for row in report_result["validation_table"].itertuples(index=False)
    ]
    test_metric_rows = [
        (row.model_name, row.test_metrics)
        for row in report_result["final_test_table"].itertuples(index=False)
    ]

    visual_paths["default_rate_trend"] = plot_default_rate_trend(
        prepared_df,
        output_dir / "default_rate_trend.png",
        show=show,
    )
    visual_paths["validation_roc_comparison"] = plot_roc_comparison(
        validation_metric_rows,
        title="Validation ROC Comparison",
        output_path=output_dir / "validation_roc_comparison.png",
        show=show,
    )
    visual_paths["test_roc_comparison"] = plot_roc_comparison(
        test_metric_rows,
        title="Test ROC Comparison",
        output_path=output_dir / "test_roc_comparison.png",
        show=show,
    )
    visual_paths["validation_calibration_comparison"] = plot_calibration_comparison(
        validation_metric_rows,
        title="Validation Reliability Diagram",
        output_path=output_dir / "validation_calibration_comparison.png",
        show=show,
    )
    visual_paths["test_calibration_comparison"] = plot_calibration_comparison(
        test_metric_rows,
        title="Test Reliability Diagram",
        output_path=output_dir / "test_calibration_comparison.png",
        show=show,
    )
    visual_paths["champion_validation_ks"] = plot_ks_curve(
        report_result["validation_predictions_by_model"][champion_key],
        title=f"{champion_name} Validation KS Curve",
        output_path=output_dir / "champion_validation_ks.png",
        show=show,
    )
    visual_paths["champion_test_ks"] = plot_ks_curve(
        report_result["test_predictions_by_model"][champion_key],
        title=f"{champion_name} Test KS Curve",
        output_path=output_dir / "champion_test_ks.png",
        show=show,
    )
    visual_paths["champion_score_distribution"] = plot_score_distribution(
        report_result["test_predictions_by_model"][champion_key],
        title=f"{champion_name} Test Score Distribution",
        output_path=output_dir / "champion_score_distribution.png",
        show=show,
    )

    xgboost_table = report_result["importance_tables"].get("xgboost")
    if xgboost_table is not None and not xgboost_table.empty:
        visual_paths["xgboost_feature_importance"] = plot_horizontal_bar(
            xgboost_table,
            label_column="feature",
            value_column="importance",
            title="XGBoost Feature Importance",
            output_path=output_dir / "xgboost_feature_importance.png",
            show=show,
        )

    logistic_table = report_result["importance_tables"].get("logistic_regression")
    if logistic_table is not None and not logistic_table.empty:
        visual_paths["logistic_coefficients"] = plot_horizontal_bar(
            logistic_table.sort_values("coefficient"),
            label_column="feature",
            value_column="coefficient",
            title="Logistic Regression Coefficients",
            output_path=output_dir / "logistic_coefficients.png",
            show=show,
            center_line=True,
        )

    if feature_search_result is not None:
        feature_search_table = feature_search_result.get("feature_search_table", pd.DataFrame())
        if not feature_search_table.empty:
            visual_paths["feature_search_deltas"] = plot_feature_search_deltas(
                feature_search_table,
                output_path=output_dir / "feature_search_deltas.png",
                show=show,
            )

    return visual_paths


def export_loss_reserve_visuals(loss_result, output_dir, show=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    workflow_df = loss_result["workflow_df"].copy()
    charged_off_proxy = loss_result["charged_off_proxy"].copy()
    resolved_holdout = loss_result["resolved_holdout"].copy()
    active_snapshot = loss_result["active_snapshot"].copy()
    portfolio_summary = loss_result["portfolio_summary"].copy()
    segment_summary = loss_result["segment_summary"].copy()
    hazard_result = loss_result["hazard_result"]

    charged_holdout = resolved_holdout[
        (resolved_holdout["loan_status"] == "Charged Off")
        & resolved_holdout["lgd_proxy"].notna()
    ].copy()

    diagnostics = {
        "lgd_diagnostics": build_lgd_diagnostics(charged_holdout),
        "ead_summary": build_ead_summary(charged_off_proxy, resolved_holdout, active_snapshot),
        "recovery_summary": build_recovery_summary(charged_off_proxy),
        "el_concentration": pd.concat(
            [
                build_el_concentration_summary(
                    resolved_holdout,
                    analysis_scope="resolved_test",
                    pd_measure="lifetime",
                    el_column="expected_loss_lifetime",
                ),
                build_el_concentration_summary(
                    active_snapshot,
                    analysis_scope="active_snapshot",
                    pd_measure="lifetime",
                    el_column="lifetime_expected_loss",
                ),
            ],
            ignore_index=True,
        ),
    }

    visual_paths = {}
    if not charged_holdout.empty:
        visual_paths["lgd_actual_vs_expected"] = plot_lgd_actual_vs_expected(
            charged_holdout,
            output_dir / "lgd_actual_vs_expected.png",
            show=show,
        )
        visual_paths["lgd_error_histogram"] = plot_lgd_error_histogram(
            charged_holdout,
            output_dir / "lgd_error_histogram.png",
            show=show,
        )
    visual_paths["lgd_by_grade"] = plot_metric_boxplot(
        charged_off_proxy,
        category_column="grade",
        value_column="lgd_proxy",
        title="LGD Distribution By Grade",
        ylabel="LGD Proxy",
        output_path=output_dir / "lgd_by_grade.png",
        show=show,
    )
    visual_paths["ead_distribution"] = plot_ead_distribution(
        charged_off_proxy,
        active_snapshot,
        output_dir / "ead_distribution.png",
        show=show,
    )
    visual_paths["ead_by_grade"] = plot_metric_boxplot(
        charged_off_proxy,
        category_column="grade",
        value_column="ead_proxy",
        title="Charged-Off EAD Proxy By Grade",
        ylabel="EAD Proxy",
        output_path=output_dir / "ead_by_grade.png",
        show=show,
        log_scale=True,
    )
    visual_paths["recovery_rate_distribution"] = plot_distribution(
        charged_off_proxy["recovery_rate_proxy"],
        title="Recovery Rate Distribution",
        xlabel="Recovery Rate",
        output_path=output_dir / "recovery_rate_distribution.png",
        show=show,
    )
    visual_paths["lgd_lookup_source"] = plot_lookup_source_usage(
        loss_result["workflow_result"]["lgd_lookup_source_table"],
        output_dir / "lgd_lookup_source.png",
        show=show,
    )
    visual_paths["hazard_vintage_calibration"] = plot_hazard_vintage_calibration(
        loss_result["workflow_result"]["hazard_validation_vintage"],
        loss_result["workflow_result"]["hazard_test_vintage"],
        output_dir / "hazard_vintage_calibration.png",
        show=show,
    )
    visual_paths["monthly_hazard_profile"] = plot_monthly_hazard_profile(
        hazard_result["panel_counts"],
        output_dir / "monthly_hazard_profile.png",
        show=show,
    )
    visual_paths["active_pd_compare"] = plot_pd_comparison(
        active_snapshot,
        output_dir / "active_pd_compare.png",
        show=show,
    )
    visual_paths["remaining_term_vs_lifetime_pd"] = plot_remaining_term_profile(
        active_snapshot,
        output_dir / "remaining_term_vs_lifetime_pd.png",
        show=show,
    )
    visual_paths["portfolio_el_comparison"] = plot_portfolio_expected_loss(
        portfolio_summary,
        output_dir / "portfolio_el_comparison.png",
        show=show,
    )
    visual_paths["predicted_vs_actual_loss"] = plot_predicted_vs_actual_loss(
        portfolio_summary,
        output_dir / "predicted_vs_actual_loss.png",
        show=show,
    )
    visual_paths["el_component_summary"] = plot_expected_loss_components(
        portfolio_summary,
        output_dir / "el_component_summary.png",
        show=show,
    )
    visual_paths["top_expected_loss_loans"] = plot_top_expected_loss_loans(
        active_snapshot,
        output_dir / "top_expected_loss_loans.png",
        show=show,
    )
    visual_paths["el_concentration_curve"] = plot_concentration_curve(
        active_snapshot,
        el_column="lifetime_expected_loss",
        title="Active Snapshot Lifetime EL Concentration",
        output_path=output_dir / "el_concentration_curve.png",
        show=show,
    )
    visual_paths["segment_el_by_grade"] = plot_segment_metric_by_scope(
        segment_summary,
        segment_type="grade",
        metric_column="total_el",
        pd_measure="lifetime",
        title="Lifetime Expected Loss By Grade",
        ylabel="Total Expected Loss",
        output_path=output_dir / "segment_el_by_grade.png",
        show=show,
    )
    visual_paths["segment_share_by_purpose"] = plot_segment_metric_by_scope(
        segment_summary,
        segment_type="purpose_group",
        metric_column="portfolio_el_share",
        pd_measure="lifetime",
        title="Lifetime EL Share By Purpose Group",
        ylabel="Portfolio EL Share",
        output_path=output_dir / "segment_share_by_purpose.png",
        show=show,
    )
    visual_paths["vintage_el_trend"] = plot_segment_metric_by_scope(
        segment_summary,
        segment_type="issue_year_quarter",
        metric_column="total_el",
        pd_measure="lifetime",
        title="Lifetime Expected Loss Trend By Origination Quarter",
        ylabel="Total Expected Loss",
        output_path=output_dir / "vintage_el_trend.png",
        show=show,
        as_line=True,
    )
    visual_paths["grade_term_heatmap"] = plot_heatmap(
        active_snapshot,
        row_column="grade",
        column_column="term_months",
        value_column="lifetime_expected_loss",
        aggfunc="mean",
        title="Active Snapshot Mean Lifetime EL: Grade x Term",
        output_path=output_dir / "grade_term_heatmap.png",
        show=show,
    )
    visual_paths["grade_fico_heatmap"] = plot_heatmap(
        resolved_holdout,
        row_column="grade",
        column_column="fico_bucket",
        value_column="pd_12m_fixed_horizon",
        aggfunc="mean",
        title="Resolved Holdout Mean 12M PD: Grade x FICO Bucket",
        output_path=output_dir / "grade_fico_heatmap.png",
        show=show,
    )

    return {"paths": visual_paths, "tables": diagnostics}


def build_lgd_diagnostics(charged_holdout):
    if charged_holdout.empty:
        return pd.DataFrame()
    error = charged_holdout["expected_lgd"] - charged_holdout["lgd_proxy"]
    correlation = charged_holdout[["expected_lgd", "lgd_proxy"]].corr().iloc[0, 1]
    return pd.DataFrame(
        [
            {
                "rows": int(len(charged_holdout)),
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "bias": float(np.mean(error)),
                "correlation": float(correlation) if pd.notna(correlation) else np.nan,
                "avg_expected_lgd": float(charged_holdout["expected_lgd"].mean()),
                "avg_actual_lgd": float(charged_holdout["lgd_proxy"].mean()),
            }
        ]
    )


def build_ead_summary(charged_off_proxy, resolved_holdout, active_snapshot):
    rows = []
    for scope_name, series in [
        ("charged_off_ead_proxy", charged_off_proxy["ead_proxy"]),
        ("resolved_holdout_ead_reference", resolved_holdout["ead_reference"]),
        ("active_snapshot_ead_current", active_snapshot["ead_current"]),
    ]:
        valid = pd.to_numeric(series, errors="coerce").dropna()
        rows.append(
            {
                "scope": scope_name,
                "rows": int(len(valid)),
                "mean": float(valid.mean()),
                "median": float(valid.median()),
                "p90": float(valid.quantile(0.90)),
                "max": float(valid.max()),
            }
        )
    return pd.DataFrame(rows)


def build_recovery_summary(charged_off_proxy):
    recovery = charged_off_proxy["recovery_rate_proxy"].dropna()
    return pd.DataFrame(
        [
            {
                "rows": int(len(recovery)),
                "mean_recovery_rate": float(recovery.mean()),
                "median_recovery_rate": float(recovery.median()),
                "p90_recovery_rate": float(recovery.quantile(0.90)),
                "mean_net_recoveries": float(charged_off_proxy["net_recoveries"].mean()),
            }
        ]
    )


def build_el_concentration_summary(df, analysis_scope, pd_measure, el_column):
    working = df[[el_column]].copy()
    working[el_column] = pd.to_numeric(working[el_column], errors="coerce").fillna(0)
    working = working.sort_values(el_column, ascending=False).reset_index(drop=True)
    total = float(working[el_column].sum())
    if total <= 0 or working.empty:
        return pd.DataFrame()

    rows = []
    for top_n in [10, 100, 1000]:
        top_share = float(working[el_column].head(min(top_n, len(working))).sum() / total)
        rows.append(
            {
                "analysis_scope": analysis_scope,
                "pd_measure": pd_measure,
                "top_n": top_n,
                "top_el_share": top_share,
            }
        )
    return pd.DataFrame(rows)


def plot_default_rate_trend(df, output_path, show=False):
    trend = (
        df.assign(issue_year_quarter=df["issue_date"].dt.to_period("Q").astype(str))
        .groupby("issue_year_quarter", as_index=False)
        .agg(default_rate=("default", "mean"), loan_count=("default", "size"))
    )
    trend["period"] = pd.PeriodIndex(trend["issue_year_quarter"], freq="Q")
    trend = trend.sort_values("period").reset_index(drop=True)
    trend["rolling_default_rate"] = trend["default_rate"].rolling(4, min_periods=1).mean()
    low_volume_threshold = max(500, int(trend["loan_count"].quantile(0.20)))
    trend["is_low_volume"] = trend["loan_count"] < low_volume_threshold

    x_positions = np.arange(len(trend))
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(12, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1]},
    )

    ax1.plot(
        x_positions,
        trend["default_rate"],
        color="#9ecae1",
        linewidth=1.6,
        alpha=0.95,
        label="Quarterly observed rate",
    )
    ax1.plot(
        x_positions,
        trend["rolling_default_rate"],
        color="#d95f0e",
        linewidth=2.2,
        label="4-quarter rolling average",
    )

    stable_mask = ~trend["is_low_volume"]
    if stable_mask.any():
        ax1.scatter(
            x_positions[stable_mask],
            trend.loc[stable_mask, "default_rate"],
            s=24,
            color="#1f77b4",
            zorder=3,
            label="Adequate-volume vintages",
        )
    if trend["is_low_volume"].any():
        ax1.scatter(
            x_positions[trend["is_low_volume"]],
            trend.loc[trend["is_low_volume"], "default_rate"],
            s=28,
            facecolors="white",
            edgecolors="gray",
            linewidth=1.2,
            zorder=4,
            label=f"Low-volume vintages (<{low_volume_threshold:,} loans)",
        )

    ax1.set_title("Observed Default Rate By Origination Quarter")
    ax1.set_ylabel("Default Rate")
    ax1.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax1.grid(alpha=0.2)
    _dedupe_legend(ax1, loc="upper left")

    bar_colors = np.where(trend["is_low_volume"], "#c7c7c7", "#ffbb78")
    ax2.bar(x_positions, trend["loan_count"], color=bar_colors, alpha=0.85)
    ax2.axhline(low_volume_threshold, color="gray", linestyle="--", linewidth=1)
    ax2.set_ylabel("Loan Count")
    ax2.set_xlabel("Origination Quarter")
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value/1000:.0f}k"))
    ax2.grid(alpha=0.2, axis="y")
    _set_categorical_ticks(ax2, x_positions, trend["issue_year_quarter"], max_ticks=16)
    return _finalize_figure(fig, output_path, show)


def plot_roc_comparison(metric_rows, title, output_path, show=False):
    fig, ax = plt.subplots(figsize=(7, 5))
    for model_name, metrics in metric_rows:
        roc_data = metrics["roc_curve"]
        ax.plot(roc_data["fpr"], roc_data["tpr"], label=f"{model_name} (AUC={metrics['AUC']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_title(title)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    ax.grid(alpha=0.2)
    return _finalize_figure(fig, output_path, show)


def plot_calibration_comparison(metric_rows, title, output_path, show=False):
    fig, ax = plt.subplots(figsize=(7, 5))
    max_bin_count = 1
    for _, metrics in metric_rows:
        calibration = metrics["calibration_curve"]
        if len(calibration.get("bin_count", [])):
            max_bin_count = max(max_bin_count, int(np.max(calibration["bin_count"])))

    for model_name, metrics in metric_rows:
        calibration = metrics["calibration_curve"]
        point_sizes = 40 + 180 * (
            calibration.get("bin_count", np.ones_like(calibration["prob_pred"])) / max_bin_count
        )
        ax.plot(
            calibration["prob_pred"],
            calibration["prob_true"],
            marker="o",
            label=model_name,
        )
        ax.scatter(
            calibration["prob_pred"],
            calibration["prob_true"],
            s=point_sizes,
            alpha=0.65,
        )
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_title(title)
    ax.set_xlabel("Predicted PD")
    ax.set_ylabel("Observed Rate")
    ax.text(
        0.98,
        0.04,
        "Marker size is proportional to bin count.",
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        ha="right",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.grid(alpha=0.2)
    return _finalize_figure(fig, output_path, show)


def plot_ks_curve(prediction_df, title, output_path, show=False):
    ks_df = _build_ks_curve_frame(prediction_df)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ks_df["population_share"], ks_df["event_share"], label="Event CDF")
    ax.plot(ks_df["population_share"], ks_df["non_event_share"], label="Non-event CDF")
    ax.plot(ks_df["population_share"], ks_df["ks_gap"], label="KS Gap", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Population Share")
    ax.set_ylabel("Cumulative Share")
    ax.legend()
    ax.grid(alpha=0.2)
    return _finalize_figure(fig, output_path, show)


def plot_score_distribution(prediction_df, title, output_path, show=False):
    sampled = _sample_frame(prediction_df, limit=150000)
    fig, ax = plt.subplots(figsize=(7, 5))
    for default_value, color in [(0, "#1f77b4"), (1, "#d62728")]:
        subset = sampled[sampled["actual_default"] == default_value]
        if subset.empty:
            continue
        ax.hist(
            subset["predicted_pd"],
            bins=40,
            alpha=0.5,
            density=True,
            label=f"actual_default={default_value}",
            color=color,
        )
    ax.set_title(title)
    ax.set_xlabel("Predicted PD")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(alpha=0.2)
    return _finalize_figure(fig, output_path, show)


def plot_horizontal_bar(
    df,
    label_column,
    value_column,
    title,
    output_path,
    show=False,
    center_line=False,
):
    display_df = df.copy()
    plot_value_column = value_column
    xlabel = value_column.replace("_", " ").title()

    if center_line:
        display_df["_abs_value"] = display_df[value_column].abs()
        display_df = display_df.nlargest(16, "_abs_value").sort_values(value_column)
    elif value_column == "importance":
        total_importance = float(display_df[value_column].sum())
        if total_importance > 0:
            plot_value_column = "relative_importance"
            display_df[plot_value_column] = display_df[value_column] / total_importance
            xlabel = "Relative Importance Share"
        display_df = display_df.nlargest(15, plot_value_column).sort_values(plot_value_column)
    else:
        display_df = display_df.nlargest(15, value_column).sort_values(value_column)

    fig, ax = plt.subplots(figsize=(8, max(4, len(display_df) * 0.35)))
    ax.barh(display_df[label_column], display_df[plot_value_column], color="#1f77b4")
    if center_line:
        ax.axvline(0, color="black", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    if plot_value_column == "relative_importance":
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.grid(alpha=0.2, axis="x")
    return _finalize_figure(fig, output_path, show)


def plot_feature_search_deltas(feature_search_table, output_path, show=False):
    display_df = feature_search_table.copy()
    display_df["_overall_change"] = (
        display_df["delta_auc"].abs()
        + display_df["delta_ks"].abs()
        + display_df["delta_brier"].abs()
    )
    display_df = display_df.nlargest(8, "_overall_change").copy()
    display_df = display_df.sort_values("_overall_change", ascending=True)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    metric_specs = [
        ("delta_auc", "Delta AUC"),
        ("delta_ks", "Delta KS"),
        ("delta_brier", "Delta Brier"),
    ]
    for ax, (column, title) in zip(axes, metric_specs):
        ax.barh(display_df["feature_group"], display_df[column], color="#1f77b4")
        ax.axvspan(-0.001, 0.001, color="#f0f0f0", alpha=0.9)
        ax.axvline(0, color="black", linewidth=1)
        ax.set_title(title)
        for _, row in display_df.iterrows():
            value = float(row[column])
            text_offset = 0.00012 if value >= 0 else -0.00012
            alignment = "left" if value >= 0 else "right"
            ax.text(
                value + text_offset,
                row["feature_group"],
                f"{value:+.4f}",
                va="center",
                ha=alignment,
                fontsize=8,
            )
        ax.grid(alpha=0.2, axis="x")
        ax.xaxis.set_major_locator(MaxNLocator(5))
    fig.suptitle("Feature Discovery Impact (Validation Deltas)")
    if not display_df.empty and display_df["delta_auc"].abs().max() < 0.001:
        fig.text(
            0.5,
            0.01,
            "AUC changes stay below 0.001, so omitted feature groups are directionally informative but not materially additive.",
            ha="center",
            fontsize=9,
        )
    fig.tight_layout()
    return _finalize_figure(fig, output_path, show)


def plot_lgd_actual_vs_expected(df, output_path, show=False):
    sampled = _sample_frame(df, limit=12000)
    fig, ax = plt.subplots(figsize=(6, 6))
    hexbin = ax.hexbin(
        sampled["lgd_proxy"],
        sampled["expected_lgd"],
        gridsize=35,
        cmap="Blues",
        mincnt=1,
    )
    fig.colorbar(hexbin, ax=ax, label="Loan Count")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_title("Expected LGD vs Actual LGD")
    ax.set_xlabel("Actual LGD Proxy")
    ax.set_ylabel("Expected LGD")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.grid(alpha=0.2)
    return _finalize_figure(fig, output_path, show)


def plot_lgd_error_histogram(df, output_path, show=False):
    error = df["expected_lgd"] - df["lgd_proxy"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(error, bins=40, color="#1f77b4", alpha=0.75)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("LGD Error Distribution")
    ax.set_xlabel("Expected LGD - Actual LGD")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.2)
    return _finalize_figure(fig, output_path, show)


def plot_metric_boxplot(
    df,
    category_column,
    value_column,
    title,
    ylabel,
    output_path,
    show=False,
    log_scale=False,
):
    display_df = df[[category_column, value_column]].dropna().copy()
    categories = sorted(display_df[category_column].astype(str).unique())
    values = [
        display_df.loc[display_df[category_column].astype(str) == category, value_column].to_numpy()
        for category in categories
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(values, labels=categories, showfliers=False)
    if log_scale:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(alpha=0.2, axis="y")
    return _finalize_figure(fig, output_path, show)


def plot_distribution(series, title, xlabel, output_path, show=False):
    values = pd.to_numeric(series, errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(values, bins=40, color="#1f77b4", alpha=0.75)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(alpha=0.2)
    return _finalize_figure(fig, output_path, show)


def plot_ead_distribution(charged_off_proxy, active_snapshot, output_path, show=False):
    charged_sample = _sample_series(charged_off_proxy["ead_proxy"], limit=150000)
    active_sample = _sample_series(active_snapshot["ead_current"], limit=150000)
    positive_values = np.concatenate(
        [
            charged_sample[charged_sample > 0].to_numpy(),
            active_sample[active_sample > 0].to_numpy(),
        ]
    )
    if len(positive_values) == 0:
        return None

    bins = np.geomspace(max(positive_values.min(), 1), positive_values.max(), 40)
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(8, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.2]},
    )

    series_specs = [
        (charged_sample[charged_sample > 0], "Charged-Off EAD Proxy", "#1f77b4"),
        (active_sample[active_sample > 0], "Active EAD Current", "#ff7f0e"),
    ]
    for values, label, color in series_specs:
        ax1.hist(
            values,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2,
            label=label,
            color=color,
        )
        ax1.axvline(values.median(), color=color, linestyle="--", linewidth=1, alpha=0.7)

        sorted_values = np.sort(values.to_numpy())
        ecdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
        ax2.plot(sorted_values, ecdf, label=label, color=color, linewidth=2)

    ax1.set_xscale("log")
    ax1.set_title("EAD Distribution Comparison")
    ax1.set_ylabel("Density")
    ax1.legend()
    ax1.grid(alpha=0.2)

    ax2.set_xscale("log")
    ax2.set_xlabel("Exposure At Default / Current Exposure")
    ax2.set_ylabel("ECDF")
    ax2.xaxis.set_major_formatter(FuncFormatter(_format_compact_currency))
    ax2.grid(alpha=0.2)
    return _finalize_figure(fig, output_path, show)


def plot_lookup_source_usage(source_table, output_path, show=False):
    display_df = source_table.copy()
    display_df["scope_source"] = display_df["analysis_scope"] + ": " + display_df["lgd_lookup_source"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(display_df["scope_source"], display_df["loan_count"], color="#1f77b4")
    ax.set_title("LGD Lookup Source Usage")
    ax.set_ylabel("Loan Count")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(alpha=0.2, axis="y")
    return _finalize_figure(fig, output_path, show)


def plot_hazard_vintage_calibration(validation_table, test_table, output_path, show=False):
    combined = pd.concat([validation_table, test_table], ignore_index=True)
    combined = combined.sort_values(["split_name", "issue_year"]).reset_index(drop=True)
    year_order = sorted(combined["issue_year"].dropna().unique())
    x_lookup = {year: index for index, year in enumerate(year_order)}
    split_offsets = {"validation": -0.10, "test": 0.10}

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, actual_col, predicted_col, title in [
        (axes[0], "actual_default_rate", "predicted_lifetime_pd", "Lifetime PD Calibration By Vintage"),
        (axes[1], "actual_default_12m_rate", "predicted_12m_pd", "12-Month PD Calibration By Vintage"),
    ]:
        rate_values = combined[[actual_col, predicted_col]].to_numpy().astype(float).ravel()
        y_min = max(0, float(np.nanmin(rate_values)) - 0.01)
        y_max = float(np.nanmax(rate_values)) + 0.03
        for split_name, split_df in combined.groupby("split_name"):
            split_df = split_df.sort_values("issue_year")
            x_positions = np.array(
                [x_lookup[year] + split_offsets.get(split_name, 0.0) for year in split_df["issue_year"]]
            )
            size_scale = split_df["loan_count"] / max(split_df["loan_count"].max(), 1)
            point_sizes = 60 + 220 * size_scale
            color = SCOPE_COLORS.get(split_name, "#1f77b4")

            ax.scatter(
                x_positions - 0.03,
                split_df[actual_col],
                s=point_sizes,
                color=color,
                marker="o",
                alpha=0.85,
                label=f"{_scope_label(split_name)} actual",
            )
            ax.scatter(
                x_positions + 0.03,
                split_df[predicted_col],
                s=point_sizes,
                color=color,
                marker="s",
                alpha=0.65,
                label=f"{_scope_label(split_name)} predicted",
            )
            for x_value, (_, row) in zip(x_positions, split_df.iterrows()):
                ax.plot(
                    [x_value - 0.03, x_value + 0.03],
                    [row[actual_col], row[predicted_col]],
                    color=color,
                    alpha=0.5,
                    linewidth=1.2,
                )
                ax.text(
                    x_value,
                    min(max(row[actual_col], row[predicted_col]) + 0.008, y_max - 0.004),
                    _format_count_short(row["loan_count"]),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#555555",
                )
        ax.set_title(title)
        ax.set_xlabel("Issue Year")
        ax.set_ylabel("Rate")
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(np.arange(len(year_order)))
        ax.set_xticklabels([str(year) for year in year_order])
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.grid(alpha=0.2)
        ax.text(
            0.02,
            0.96,
            "Point size is proportional to vintage loan count.\nNo line interpolation is used because vintage coverage is sparse.",
            transform=ax.transAxes,
            fontsize=8.5,
            va="top",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
    _dedupe_legend(axes[1], loc="upper right", fontsize=8)
    fig.tight_layout()
    return _finalize_figure(fig, output_path, show)


def plot_monthly_hazard_profile(panel_counts, output_path, show=False):
    display_df = panel_counts.copy()
    display_df["month_bucket"] = pd.Categorical(
        display_df["month_bucket"],
        categories=MONTH_BUCKET_ORDER,
        ordered=True,
    )
    profile = (
        display_df.groupby(
            ["split_label", "month_bucket"],
            observed=True,
        )
        .agg(exposure_count=("exposure_count", "sum"), event_count=("event_count", "sum"))
        .reset_index()
    )
    profile["hazard_rate"] = profile["event_count"] / profile["exposure_count"]
    fig, ax = plt.subplots(figsize=(8, 4))
    for split_name, split_df in profile.groupby("split_label"):
        split_df = split_df.sort_values("month_bucket")
        ax.plot(split_df["month_bucket"].astype(str), split_df["hazard_rate"], marker="o", label=split_name)
    ax.set_title("Monthly Hazard Rate By Month-On-Book Bucket")
    ax.set_xlabel("Month Bucket")
    ax.set_ylabel("Hazard Rate")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.grid(alpha=0.2)
    ax.legend()
    return _finalize_figure(fig, output_path, show)


def plot_pd_comparison(active_snapshot, output_path, show=False):
    sampled = _sample_frame(active_snapshot, limit=120000)
    fig, ax = plt.subplots(figsize=(6, 5))
    hexbin = ax.hexbin(
        sampled["pd_12m_fixed_horizon"],
        sampled["lifetime_pd_hazard"],
        gridsize=40,
        cmap="Blues",
        mincnt=1,
    )
    fig.colorbar(hexbin, ax=ax, label="Loan Count")
    ax.set_title("12-Month PD vs Lifetime PD")
    ax.set_xlabel("12-Month Fixed-Horizon PD")
    ax.set_ylabel("Lifetime Hazard PD")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.grid(alpha=0.1)
    return _finalize_figure(fig, output_path, show)


def plot_remaining_term_profile(active_snapshot, output_path, show=False):
    profile = (
        active_snapshot.groupby("remaining_term", as_index=False)
        .agg(
            avg_lifetime_pd=("lifetime_pd_hazard", "mean"),
            loan_count=("sample_id", "size"),
        )
        .sort_values("remaining_term")
    )
    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax1.plot(profile["remaining_term"], profile["avg_lifetime_pd"], marker="o", color="#1f77b4")
    ax1.set_title("Remaining Term vs Lifetime PD")
    ax1.set_xlabel("Remaining Term")
    ax1.set_ylabel("Average Lifetime PD")
    ax1.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax1.grid(alpha=0.2)
    ax2 = ax1.twinx()
    ax2.bar(profile["remaining_term"], profile["loan_count"], alpha=0.15, color="#ff7f0e")
    ax2.set_ylabel("Loan Count")
    return _finalize_figure(fig, output_path, show)


def plot_portfolio_expected_loss(portfolio_summary, output_path, show=False):
    display_df = portfolio_summary.copy()
    if display_df.empty:
        return None

    display_df["el_rate"] = display_df["total_el"] / display_df["funded_amount"].replace(0, np.nan)
    display_df["el_per_loan"] = display_df["total_el"] / display_df["loan_count"].replace(0, np.nan)

    scopes = [scope for scope in SCOPE_ORDER if scope in display_df["analysis_scope"].unique()]
    metric_specs = [
        ("total_el", "Total Expected Loss", "currency"),
        ("el_rate", "Expected Loss Rate", "percent"),
        ("el_per_loan", "Expected Loss Per Loan", "currency"),
    ]

    fig, axes = plt.subplots(len(scopes), len(metric_specs), figsize=(14, 6), squeeze=False)
    for row_index, scope in enumerate(scopes):
        scope_df = (
            display_df[display_df["analysis_scope"] == scope]
            .set_index("pd_measure")
            .reindex(PD_MEASURE_ORDER)
        )
        for col_index, (column, title, axis_type) in enumerate(metric_specs):
            ax = axes[row_index, col_index]
            ax.bar(
                [PD_MEASURE_LABELS[key] for key in scope_df.index],
                scope_df[column].fillna(0),
                color=["#4c78a8", "#f58518"],
            )
            ax.set_title(f"{_scope_label(scope)}: {title}")
            if axis_type == "currency":
                ax.yaxis.set_major_formatter(FuncFormatter(_format_compact_currency))
            else:
                ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
            ax.grid(alpha=0.2, axis="y")
    return _finalize_figure(fig, output_path, show)


def plot_predicted_vs_actual_loss(portfolio_summary, output_path, show=False):
    resolved_df = portfolio_summary[portfolio_summary["analysis_scope"] == "resolved_test"].copy()
    if resolved_df.empty:
        return None

    matched_row = resolved_df[resolved_df["pd_measure"] == "12m"].copy()
    if matched_row.empty:
        matched_row = resolved_df.head(1).copy()
    matched_row = matched_row.iloc[0]

    predicted_loss_rate = float(matched_row["total_el"] / matched_row["funded_amount"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].bar(
        ["Predicted 12M EL", "Actual Net Loss"],
        [matched_row["total_el"], matched_row["actual_loss_amount"]],
        color=["#4c78a8", "#e45756"],
    )
    axes[0].set_title("Resolved Holdout: Loss Amount")
    axes[0].set_ylabel("Loss Amount")
    axes[0].yaxis.set_major_formatter(FuncFormatter(_format_compact_currency))
    axes[0].grid(alpha=0.2, axis="y")

    axes[1].bar(
        ["Predicted 12M EL Rate", "Actual Net Loss Rate"],
        [predicted_loss_rate, matched_row["actual_loss_rate"]],
        color=["#4c78a8", "#e45756"],
    )
    axes[1].set_title("Resolved Holdout: Loss Rate")
    axes[1].set_ylabel("Rate")
    axes[1].yaxis.set_major_formatter(PercentFormatter(xmax=1))
    axes[1].grid(alpha=0.2, axis="y")

    fig.suptitle("Matched-Horizon Loss Comparison")
    fig.text(
        0.5,
        0.01,
        "Lifetime reserve estimates are shown separately and are not directly compared against realized resolved-vintage loss.",
        ha="center",
        fontsize=9,
    )
    return _finalize_figure(fig, output_path, show)


def plot_expected_loss_components(portfolio_summary, output_path, show=False):
    display_df = portfolio_summary.copy()
    if display_df.empty:
        return None
    display_df["el_rate"] = display_df["total_el"] / display_df["funded_amount"].replace(0, np.nan)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    metrics = [
        ("avg_pd", "Average PD"),
        ("avg_lgd", "Average LGD"),
        ("avg_ead", "Average EAD"),
        ("el_rate", "Expected Loss Rate"),
    ]
    scopes = [scope for scope in SCOPE_ORDER if scope in display_df["analysis_scope"].unique()]
    x_positions = np.arange(len(PD_MEASURE_ORDER))
    width = 0.36

    for ax, (column, title) in zip(axes.flatten(), metrics):
        for index, scope in enumerate(scopes):
            scope_df = (
                display_df[display_df["analysis_scope"] == scope]
                .set_index("pd_measure")
                .reindex(PD_MEASURE_ORDER)
            )
            ax.bar(
                x_positions + (index - (len(scopes) - 1) / 2) * width,
                scope_df[column].fillna(0),
                width=width,
                label=_scope_label(scope),
                color=SCOPE_COLORS.get(scope, "#1f77b4"),
                alpha=0.85,
            )
        ax.set_title(title)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([PD_MEASURE_LABELS[key] for key in PD_MEASURE_ORDER])
        if column in {"avg_pd", "avg_lgd", "el_rate"}:
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        else:
            ax.yaxis.set_major_formatter(FuncFormatter(_format_compact_currency))
        ax.grid(alpha=0.2, axis="y")
    _dedupe_legend(axes[0, 0], loc="upper left")
    fig.tight_layout()
    return _finalize_figure(fig, output_path, show)


def plot_top_expected_loss_loans(active_snapshot, output_path, show=False):
    required_columns = [
        "sample_id",
        "grade",
        "purpose_group",
        "lifetime_expected_loss",
        "lifetime_pd_hazard",
        "expected_lgd",
        "ead_current",
    ]
    display_df = active_snapshot.nlargest(15, "lifetime_expected_loss")[required_columns].copy()
    display_df["loan_label"] = display_df.apply(
        lambda row: (
            f"{int(row['sample_id'])} | {row.get('grade', 'NA')} | "
            f"{str(row.get('purpose_group', 'other')).replace('_', ' ')[:16]}"
        ),
        axis=1,
    )
    display_df = display_df.sort_values("lifetime_expected_loss")

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(display_df["loan_label"], display_df["lifetime_expected_loss"], color="#1f77b4")
    ax.set_title("Top Active Loans By Lifetime Expected Loss")
    ax.set_xlabel("Lifetime Expected Loss")
    ax.xaxis.set_major_formatter(FuncFormatter(_format_compact_currency))
    for _, row in display_df.iterrows():
        annotation = (
            f"PD {row['lifetime_pd_hazard']:.0%} | "
            f"LGD {row['expected_lgd']:.0%} | "
            f"EAD {_format_compact_currency(row['ead_current'], None)}"
        )
        ax.text(
            row["lifetime_expected_loss"] * 1.01,
            row["loan_label"],
            annotation,
            va="center",
            fontsize=8,
        )
    ax.grid(alpha=0.2, axis="x")
    return _finalize_figure(fig, output_path, show)


def plot_concentration_curve(df, el_column, title, output_path, show=False):
    working = df[[el_column]].copy()
    working[el_column] = pd.to_numeric(working[el_column], errors="coerce").fillna(0)
    working = working.sort_values(el_column, ascending=False).reset_index(drop=True)
    if working.empty:
        return None
    total_el = working[el_column].sum()
    working["population_share"] = np.arange(1, len(working) + 1) / len(working)
    working["el_share"] = working[el_column].cumsum() / total_el
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(working["population_share"], working["el_share"], color="#1f77b4")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_title(title)
    ax.set_xlabel("Population Share")
    ax.set_ylabel("Cumulative EL Share")
    ax.grid(alpha=0.2)
    return _finalize_figure(fig, output_path, show)


def plot_segment_metric_by_scope(
    segment_summary,
    segment_type,
    metric_column,
    pd_measure,
    title,
    ylabel,
    output_path,
    show=False,
    as_line=False,
):
    display_df = segment_summary[
        (segment_summary["segment_type"] == segment_type)
        & (segment_summary["pd_measure"] == pd_measure)
    ].copy()
    if display_df.empty:
        return None

    ordered_segments = _sort_segment_values(display_df["segment_value"].dropna().unique(), segment_type)
    scope_order = [
        scope
        for scope in SCOPE_ORDER
        if scope in display_df["analysis_scope"].dropna().unique()
    ]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for analysis_scope, scope_df in display_df.groupby("analysis_scope"):
        scope_df = scope_df.set_index("segment_value").reindex(ordered_segments).reset_index()
        if as_line:
            color = SCOPE_COLORS.get(analysis_scope, "#1f77b4")
            x_positions = np.arange(len(scope_df))
            ax.plot(
                x_positions,
                scope_df[metric_column],
                marker="o",
                label=_scope_label(analysis_scope),
                color=color,
                linewidth=2,
            )
        else:
            continue

    if not as_line:
        x_positions = np.arange(len(ordered_segments))
        width = 0.36
        for index, analysis_scope in enumerate(scope_order):
            scope_df = (
                display_df[display_df["analysis_scope"] == analysis_scope]
                .set_index("segment_value")
                .reindex(ordered_segments)
            )
            ax.bar(
                x_positions + (index - (len(scope_order) - 1) / 2) * width,
                scope_df[metric_column].fillna(0),
                width=width,
                alpha=0.85,
                label=_scope_label(analysis_scope),
                color=SCOPE_COLORS.get(analysis_scope, "#1f77b4"),
            )
        ax.set_xticks(x_positions)
    else:
        ax.set_xticks(np.arange(len(ordered_segments)))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    _set_categorical_ticks(ax, np.arange(len(ordered_segments)), ordered_segments, max_ticks=14)
    if metric_column.endswith("share") or metric_column.endswith("rate") or metric_column in {
        "avg_pd",
        "avg_lgd",
    }:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    elif metric_column in {"avg_ead", "avg_el", "total_el"}:
        ax.yaxis.set_major_formatter(FuncFormatter(_format_compact_currency))
    ax.grid(alpha=0.2, axis="y")
    _dedupe_legend(ax)
    return _finalize_figure(fig, output_path, show)


def plot_heatmap(
    df,
    row_column,
    column_column,
    value_column,
    aggfunc,
    title,
    output_path,
    show=False,
):
    heatmap = pd.pivot_table(
        df,
        index=row_column,
        columns=column_column,
        values=value_column,
        aggfunc=aggfunc,
    )
    count_map = pd.pivot_table(
        df,
        index=row_column,
        columns=column_column,
        values=value_column,
        aggfunc="count",
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    image = ax.imshow(heatmap.fillna(0).to_numpy(), cmap="Blues", aspect="auto")
    fig.colorbar(image, ax=ax)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(heatmap.columns)))
    ax.set_xticklabels([str(value) for value in heatmap.columns], rotation=30)
    ax.set_yticks(np.arange(len(heatmap.index)))
    ax.set_yticklabels([str(value) for value in heatmap.index])
    value_array = heatmap.to_numpy()
    valid_values = value_array[~np.isnan(value_array)]
    threshold = np.nanmedian(valid_values) if len(valid_values) else 0
    for row_index, row_label in enumerate(heatmap.index):
        for col_index, col_label in enumerate(heatmap.columns):
            value = heatmap.loc[row_label, col_label]
            count = count_map.loc[row_label, col_label]
            if pd.isna(value):
                text = "NA"
            else:
                text = f"{_format_heatmap_value(value)}\n(n={int(count)})"
            text_color = "white" if pd.notna(value) and value >= threshold else "#1a1a1a"
            ax.text(
                col_index,
                row_index,
                text,
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
            )
    return _finalize_figure(fig, output_path, show)


def _build_ks_curve_frame(prediction_df):
    working = prediction_df[["actual_default", "predicted_pd"]].copy()
    working = working.sort_values("predicted_pd", ascending=False).reset_index(drop=True)
    working["non_event"] = 1 - working["actual_default"]
    total_events = working["actual_default"].sum()
    total_non_events = working["non_event"].sum()
    working["population_share"] = np.arange(1, len(working) + 1) / len(working)
    working["event_share"] = working["actual_default"].cumsum() / max(total_events, 1)
    working["non_event_share"] = working["non_event"].cumsum() / max(total_non_events, 1)
    working["ks_gap"] = working["event_share"] - working["non_event_share"]
    return working


def _sample_frame(df, limit):
    if len(df) <= limit:
        return df.copy()
    return df.sample(n=limit, random_state=DEFAULT_RANDOM_STATE).copy()


def _sample_series(series, limit):
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) <= limit:
        return values
    return values.sample(n=limit, random_state=DEFAULT_RANDOM_STATE)


def _scope_label(scope_name):
    return SCOPE_LABELS.get(scope_name, str(scope_name).replace("_", " ").title())


def _format_compact_currency(value, _):
    if pd.isna(value):
        return "NA"
    value = float(value)
    abs_value = abs(value)
    if abs_value >= 1_000_000_000:
        return f"${value/1_000_000_000:.1f}B"
    if abs_value >= 1_000_000:
        return f"${value/1_000_000:.1f}M"
    if abs_value >= 1_000:
        return f"${value/1_000:.1f}k"
    return f"${value:.0f}"


def _format_heatmap_value(value):
    value = float(value)
    abs_value = abs(value)
    if abs_value <= 1:
        return f"{value:.1%}"
    if abs_value >= 1_000:
        return _format_compact_currency(value, None)
    return f"{value:.2f}"


def _format_count_short(value):
    value = int(value)
    if value >= 1_000_000:
        return f"n={value/1_000_000:.1f}M"
    if value >= 1_000:
        return f"n={value/1_000:.0f}k"
    return f"n={value}"


def _sort_segment_values(values, segment_type):
    values = list(values)
    if segment_type in SEGMENT_SORT_ORDERS:
        order_lookup = {value: index for index, value in enumerate(SEGMENT_SORT_ORDERS[segment_type])}
        return sorted(values, key=lambda value: (order_lookup.get(value, len(order_lookup)), str(value)))
    if segment_type == "issue_year_quarter":
        return [str(period) for period in sorted(pd.PeriodIndex(pd.Index(values).astype(str), freq="Q"))]
    if segment_type in {"issue_year", "term_months"}:
        numeric_values = [
            value for value in pd.to_numeric(pd.Index(values), errors="coerce").tolist() if pd.notna(value)
        ]
        ordered = sorted(numeric_values)
        return [int(value) if float(value).is_integer() else float(value) for value in ordered]
    return sorted(values, key=lambda value: str(value))


def _set_categorical_ticks(ax, x_positions, labels, max_ticks=16):
    labels = [str(label).replace("_", " ") for label in labels]
    x_positions = np.asarray(x_positions)
    if len(labels) == 0:
        return
    step = max(1, int(np.ceil(len(labels) / max_ticks)))
    ax.set_xticks(x_positions[::step])
    ax.set_xticklabels(labels[::step], rotation=45, ha="right")


def _dedupe_legend(ax, **legend_kwargs):
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    deduped = {}
    for handle, label in zip(handles, labels):
        if label not in deduped:
            deduped[label] = handle
    ax.legend(deduped.values(), deduped.keys(), **legend_kwargs)


def _finalize_figure(fig, output_path, show):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return output_path.resolve()
