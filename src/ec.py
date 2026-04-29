from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd


EC_COLUMN_CANDIDATES = {
    "loan_id": ["loan_id", "sample_id", "id", "member_id"],
    "PD_12m": ["pd_12m_fixed_horizon", "pd_12m", "stage2_champion_pd_12m", "pd_12m_hazard"],
    "LGD": ["expected_lgd", "lgd_proxy", "lgd"],
    "EAD": ["ead_reference", "ead_current", "ead_proxy", "ead"],
    "loan_EL_12m": ["expected_loss_12m", "loan_EL_12m", "el_12m"],
    "funded_amnt": ["funded_amnt", "loan_amnt"],
    "grade": ["grade"],
    "term_months": ["term_months"],
    "actual_default_12m": ["actual_default_12m"],
}

REQUIRED_EC_FIELDS = ["PD_12m", "LGD", "EAD", "loan_EL_12m"]


def select_ec_columns(df):
    """
    Auto-detect column names needed by EC workflow.
    Returns a mapping from standardized EC fields to source columns.
    """
    column_map = {}
    for target_field, candidates in EC_COLUMN_CANDIDATES.items():
        column_map[target_field] = _first_existing_column(df, candidates)

    missing_required = [field for field in REQUIRED_EC_FIELDS if column_map.get(field) is None]
    if missing_required:
        raise KeyError(f"Missing required EC columns for: {missing_required}")
    return column_map


def prepare_ec_dataset(df):
    """
    Build clean EC dataset used by downstream EC calculations only.
    """
    column_map = select_ec_columns(df)
    ec_df = pd.DataFrame(index=df.index.copy())

    if column_map.get("loan_id") is not None:
        ec_df["loan_id"] = df[column_map["loan_id"]]
    else:
        ec_df["loan_id"] = np.arange(len(df))

    numeric_required = ["PD_12m", "LGD", "EAD", "loan_EL_12m"]
    for field in numeric_required:
        ec_df[field] = pd.to_numeric(df[column_map[field]], errors="coerce")

    ec_df["PD_12m"] = ec_df["PD_12m"].fillna(0).clip(lower=0, upper=1)
    ec_df["LGD"] = ec_df["LGD"].fillna(0).clip(lower=0, upper=1)
    ec_df["EAD"] = ec_df["EAD"].fillna(0).clip(lower=0)
    ec_df["loan_EL_12m"] = ec_df["loan_EL_12m"].fillna(0).clip(lower=0)

    for optional_field in ["funded_amnt", "grade", "term_months", "actual_default_12m"]:
        source_col = column_map.get(optional_field)
        if source_col is None:
            continue
        if optional_field in {"funded_amnt", "term_months", "actual_default_12m"}:
            ec_df[optional_field] = pd.to_numeric(df[source_col], errors="coerce")
        else:
            ec_df[optional_field] = df[source_col]

    if "funded_amnt" in ec_df.columns:
        ec_df["funded_amnt"] = ec_df["funded_amnt"].fillna(0).clip(lower=0)
    if "term_months" in ec_df.columns:
        ec_df["term_months"] = ec_df["term_months"].fillna(0).clip(lower=0)
    if "actual_default_12m" in ec_df.columns:
        ec_df["actual_default_12m"] = (
            ec_df["actual_default_12m"].fillna(0).clip(lower=0, upper=1).round().astype(int)
        )

    return ec_df.reset_index(drop=True)


def calculate_portfolio_el(ec_df):
    """
    Portfolio EL summary from loan-level 12m EL fields.
    """
    total_ead = float(ec_df["EAD"].sum())
    portfolio_el = float(ec_df["loan_EL_12m"].sum())
    portfolio_el_rate = float(portfolio_el / total_ead) if total_ead > 0 else 0.0

    portfolio_el_summary = {
        "loan_count": int(len(ec_df)),
        "total_EAD": total_ead,
        "portfolio_EL": portfolio_el,
        "portfolio_EL_rate": portfolio_el_rate,
        "avg_PD_12m": float(ec_df["PD_12m"].mean()) if len(ec_df) else 0.0,
        "avg_LGD": float(ec_df["LGD"].mean()) if len(ec_df) else 0.0,
        "avg_EAD": float(ec_df["EAD"].mean()) if len(ec_df) else 0.0,
        "avg_loan_EL_12m": float(ec_df["loan_EL_12m"].mean()) if len(ec_df) else 0.0,
    }
    return portfolio_el_summary


def estimate_ec_monte_carlo(
    ec_df,
    n_sim=10000,
    confidence_levels=(0.99, 0.999, 0.9999),
    seed=42,
    batch_size=200,
    auto_stop=False,
    min_sim=10000,
    check_every=5000,
    rel_tol=0.01,
    target_confidence=0.99,
    stable_checks=2,
):
    """
    Independent default Monte Carlo EC.
    """
    rng = np.random.default_rng(seed)
    pd_12m = ec_df["PD_12m"].to_numpy(dtype=float)
    loss_given_default = (ec_df["LGD"] * ec_df["EAD"]).to_numpy(dtype=float)

    _validate_mc_params(
        n_sim=n_sim,
        batch_size=batch_size,
        min_sim=min_sim,
        check_every=check_every,
        rel_tol=rel_tol,
        stable_checks=stable_checks,
    )

    n_sim = int(n_sim)
    batch_size = int(batch_size)
    min_sim = int(min_sim)
    check_every = int(check_every)
    stable_checks = int(stable_checks)

    loan_count = len(ec_df)
    total_ead = float(ec_df["EAD"].sum())
    portfolio_el = float(ec_df["loan_EL_12m"].sum())
    simulated_losses = np.empty(n_sim, dtype=float)

    sims_done = 0
    converged = False
    convergence_history = []
    stable_hit_count = 0
    next_check = max(check_every, min_sim)

    while sims_done < n_sim:
        this_batch = min(batch_size, n_sim - sims_done)
        default_matrix = rng.random((this_batch, loan_count)) < pd_12m
        batch_loss = default_matrix @ loss_given_default
        simulated_losses[sims_done : sims_done + this_batch] = batch_loss
        sims_done += this_batch

        if auto_stop and sims_done >= next_check and sims_done >= min_sim:
            current_var = float(np.quantile(simulated_losses[:sims_done], target_confidence))
            convergence_history.append(current_var)
            if len(convergence_history) >= 2:
                prev_var = convergence_history[-2]
                if abs(prev_var) > 1e-12:
                    rel_change = abs(current_var - prev_var) / abs(prev_var)
                else:
                    rel_change = abs(current_var - prev_var)
                if rel_change <= rel_tol:
                    stable_hit_count += 1
                else:
                    stable_hit_count = 0
                if stable_hit_count >= stable_checks:
                    converged = True
                    break
            next_check += check_every

    simulated_losses = simulated_losses[:sims_done]
    simulated_mean_loss = float(simulated_losses.mean()) if sims_done > 0 else 0.0
    rows = []
    for confidence_level in confidence_levels:
        var_value = float(np.quantile(simulated_losses, confidence_level))
        ec_value = float(var_value - portfolio_el)
        rows.append(
            {
                "method": "independent_mc",
                "confidence_level": float(confidence_level),
                "loan_count": int(loan_count),
                "total_EAD": total_ead,
                "portfolio_EL": portfolio_el,
                "simulated_mean_loss": simulated_mean_loss,
                "VaR": var_value,
                "EC": ec_value,
                "EC_rate": float(ec_value / total_ead) if total_ead > 0 else 0.0,
                "n_sim": int(sims_done),
                "seed": int(seed),
                "auto_stop": bool(auto_stop),
                "converged": bool(converged),
                "n_sim_target": int(n_sim),
                "min_sim": int(min_sim),
                "check_every": int(check_every),
                "rel_tol": float(rel_tol),
                "target_confidence": float(target_confidence),
                "stable_checks": int(stable_checks),
            }
        )
    return simulated_losses, pd.DataFrame(rows)


def estimate_ec_one_factor_monte_carlo(
    ec_df,
    n_sim=10000,
    confidence_levels=(0.99, 0.999, 0.9999),
    rho=0.10,
    rho_mode="constant",
    seed=42,
    batch_size=200,
    auto_stop=False,
    min_sim=10000,
    check_every=5000,
    rel_tol=0.01,
    target_confidence=0.99,
    stable_checks=2,
):
    """
    One-factor ASRF (Vasicek closed-form WCDR) EC.
    Kept function name for backward compatibility.
    """
    rho = float(rho)

    pd_12m = ec_df["PD_12m"].to_numpy(dtype=float)
    lgd = ec_df["LGD"].to_numpy(dtype=float)
    ead = ec_df["EAD"].to_numpy(dtype=float)
    loss_given_default = lgd * ead

    eps = 1e-12
    pd_12m = np.clip(pd_12m, eps, 1 - eps)
    thresholds = _norm_ppf(pd_12m)
    rho_vector = _build_asrf_rho_vector(pd_12m, rho=rho, rho_mode=rho_mode)

    loan_count = len(ec_df)
    total_ead = float(ead.sum())
    portfolio_el = float(ec_df["loan_EL_12m"].sum())
    expected_default_rate = float(pd_12m.mean())

    rows = []
    for confidence_level in confidence_levels:
        # Vasicek closed-form conditional default probability per obligor
        alpha_z = float(_norm_ppf(np.array([confidence_level]))[0])
        wcdr_vector = _norm_cdf((thresholds + np.sqrt(rho_vector) * alpha_z) / np.sqrt(1.0 - rho_vector))
        # Portfolio loss quantile proxy
        var_value = float(np.sum(wcdr_vector * loss_given_default))
        ec_value = float(var_value - portfolio_el)
        rows.append(
            {
                "method": "asrf_vasicek_closed_form",
                "confidence_level": float(confidence_level),
                "rho_mode": str(rho_mode),
                "rho": float(np.mean(rho_vector)),
                "rho_min": float(np.min(rho_vector)),
                "rho_max": float(np.max(rho_vector)),
                "expected_default_rate": expected_default_rate,
                "wcdr_mean": float(np.mean(wcdr_vector)),
                "loan_count": int(loan_count),
                "total_EAD": total_ead,
                "portfolio_EL": portfolio_el,
                "simulated_mean_loss": np.nan,
                "VaR": var_value,
                "EC": ec_value,
                "EC_rate": float(ec_value / total_ead) if total_ead > 0 else 0.0,
                "n_sim": np.nan,
                "seed": int(seed),
                "auto_stop": np.nan,
                "converged": np.nan,
                "n_sim_target": np.nan,
                "min_sim": np.nan,
                "check_every": np.nan,
                "rel_tol": np.nan,
                "target_confidence": np.nan,
                "stable_checks": np.nan,
            }
        )
    return np.array([], dtype=float), pd.DataFrame(rows)


def estimate_ec_one_factor_sensitivity(
    ec_df,
    rho_grid=(0.0, 0.05, 0.10, 0.20),
    n_sim=10000,
    confidence_levels=(0.99, 0.999, 0.9999),
    seed=42,
    batch_size=200,
    auto_stop=False,
    min_sim=10000,
    check_every=5000,
    rel_tol=0.01,
    target_confidence=0.99,
    stable_checks=2,
):
    """
    Run one-factor Monte Carlo under multiple rho values.
    """
    summaries = []
    for idx, rho in enumerate(rho_grid):
        _, summary = estimate_ec_one_factor_monte_carlo(
            ec_df=ec_df,
            n_sim=n_sim,
            confidence_levels=confidence_levels,
            rho=float(rho),
            rho_mode="constant",
            seed=int(seed) + idx,
            batch_size=batch_size,
            auto_stop=auto_stop,
            min_sim=min_sim,
            check_every=check_every,
            rel_tol=rel_tol,
            target_confidence=target_confidence,
            stable_checks=stable_checks,
        )
        summaries.append(summary)
    if not summaries:
        return pd.DataFrame()
    return pd.concat(summaries, ignore_index=True)


def realized_loss_backtest(ec_df):
    """
    Realized-loss backtest using actual_default_12m if available.
    """
    if "actual_default_12m" not in ec_df.columns:
        realized_summary = pd.DataFrame(
            [
                {
                    "loan_count": int(len(ec_df)),
                    "portfolio_EL": float(ec_df["loan_EL_12m"].sum()),
                    "realized_portfolio_loss": np.nan,
                    "difference": np.nan,
                    "difference_pct": np.nan,
                    "actual_default_rate": np.nan,
                    "avg_PD_12m": float(ec_df["PD_12m"].mean()) if len(ec_df) else np.nan,
                    "backtest_available": False,
                }
            ]
        )
        return realized_summary, None

    backtest_df = ec_df.copy()
    backtest_df["actual_default_12m"] = (
        pd.to_numeric(backtest_df["actual_default_12m"], errors="coerce")
        .fillna(0)
        .clip(lower=0, upper=1)
        .round()
        .astype(int)
    )
    backtest_df["realized_loss_12m"] = (
        backtest_df["actual_default_12m"] * backtest_df["LGD"] * backtest_df["EAD"]
    )

    realized_portfolio_loss = float(backtest_df["realized_loss_12m"].sum())
    portfolio_el = float(backtest_df["loan_EL_12m"].sum())
    difference = float(realized_portfolio_loss - portfolio_el)
    difference_pct = float(difference / portfolio_el) if portfolio_el > 0 else np.nan

    realized_summary = pd.DataFrame(
        [
            {
                "loan_count": int(len(backtest_df)),
                "portfolio_EL": portfolio_el,
                "realized_portfolio_loss": realized_portfolio_loss,
                "difference": difference,
                "difference_pct": difference_pct,
                "actual_default_rate": float(backtest_df["actual_default_12m"].mean()),
                "avg_PD_12m": float(backtest_df["PD_12m"].mean()) if len(backtest_df) else np.nan,
                "backtest_available": True,
            }
        ]
    )

    grade_summary = None
    if "grade" in backtest_df.columns:
        grade_summary = (
            backtest_df.groupby("grade", dropna=False)
            .agg(
                loan_count=("loan_id", "size"),
                avg_PD_12m=("PD_12m", "mean"),
                actual_default_rate=("actual_default_12m", "mean"),
                predicted_EL=("loan_EL_12m", "sum"),
                realized_loss=("realized_loss_12m", "sum"),
                total_EAD=("EAD", "sum"),
            )
            .reset_index()
        )
    return realized_summary, grade_summary


def create_ec_visualizations(
    simulated_losses,
    monte_carlo_summary,
    portfolio_EL,
    output_dir="outputs/figures",
    file_name="portfolio_loss_distribution_var_ec.png",
):
    """
    Plot simulated portfolio loss distribution with EL and VaR markers.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise ImportError(
            "matplotlib is required to create EC visualizations. "
            "Please install matplotlib and rerun."
        ) from exc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / file_name

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(simulated_losses, bins=60, alpha=0.75, color="#4C78A8", edgecolor="white")
    ax.axvline(float(portfolio_EL), color="#2CA02C", linestyle="--", linewidth=2, label="Portfolio EL")

    var_99 = _extract_var(monte_carlo_summary, 0.99)
    if var_99 is not None:
        ax.axvline(var_99, color="#D62728", linestyle="-", linewidth=2, label="VaR 99%")

    var_999 = _extract_var(monte_carlo_summary, 0.999)
    if var_999 is not None:
        ax.axvline(var_999, color="#9467BD", linestyle="-.", linewidth=2, label="VaR 99.9%")

    var_9999 = _extract_var(monte_carlo_summary, 0.9999)
    if var_9999 is not None:
        ax.axvline(var_9999, color="#8C564B", linestyle=":", linewidth=2, label="VaR 99.99%")

    ax.set_title("Portfolio Loss Distribution With EL And VaR")
    ax.set_xlabel("Simulated Portfolio Loss")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return str(output_path)


def save_ec_outputs(
    ec_df,
    portfolio_el_summary,
    independent_summary,
    asrf_summary,
    asrf_sensitivity_summary=None,
    method_comparison_summary=None,
    realized_summary=None,
    grade_summary=None,
    output_dir="outputs",
):
    """
    Save EC outputs to CSV files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ec_df.to_csv(output_dir / "ec_input_with_loan_el.csv", index=False)

    pd.DataFrame([portfolio_el_summary]).to_csv(output_dir / "ec_portfolio_el_summary.csv", index=False)
    independent_summary_csv = _compact_mc_summary_for_output(independent_summary)
    asrf_summary_csv = _compact_mc_summary_for_output(asrf_summary)
    independent_summary_csv.to_csv(output_dir / "ec_independent_monte_carlo_summary.csv", index=False)
    asrf_summary_csv.to_csv(output_dir / "ec_asrf_monte_carlo_summary.csv", index=False)
    independent_summary_csv.to_csv(output_dir / "ec_monte_carlo_summary.csv", index=False)
    if asrf_sensitivity_summary is not None and not asrf_sensitivity_summary.empty:
        _compact_mc_summary_for_output(asrf_sensitivity_summary).to_csv(
            output_dir / "ec_asrf_sensitivity_summary.csv", index=False
        )
    if method_comparison_summary is not None and not method_comparison_summary.empty:
        _compact_mc_summary_for_output(method_comparison_summary).to_csv(
            output_dir / "ec_method_comparison_summary.csv", index=False
        )

    if realized_summary is not None and not realized_summary.empty:
        realized_summary.to_csv(output_dir / "ec_realized_loss_backtest.csv", index=False)
    if grade_summary is not None and not grade_summary.empty:
        grade_summary.to_csv(output_dir / "ec_grade_level_backtest.csv", index=False)


def run_ec_pipeline(
    df,
    n_sim=10000,
    confidence_levels=(0.99, 0.999, 0.9999),
    seed=42,
    batch_size=200,
    auto_stop=False,
    min_sim=10000,
    check_every=5000,
    rel_tol=0.01,
    target_confidence=0.99,
    stable_checks=2,
    asrf_rho=0.10,
    asrf_rho_mode="constant",
    asrf_rho_grid=None,
    output_dir="outputs",
):
    """
    One-click EC pipeline using existing 12m EL inputs.
    Runs independent MC and ASRF one-factor MC with matching simulation settings.
    """
    independent_visualization_path = None
    asrf_visualization_path = None
    comparison_visual_paths = {}
    ec_df = prepare_ec_dataset(df)
    portfolio_el_summary = calculate_portfolio_el(ec_df)

    independent_losses, independent_summary = estimate_ec_monte_carlo(
        ec_df,
        n_sim=n_sim,
        confidence_levels=confidence_levels,
        seed=seed,
        batch_size=batch_size,
        auto_stop=auto_stop,
        min_sim=min_sim,
        check_every=check_every,
        rel_tol=rel_tol,
        target_confidence=target_confidence,
        stable_checks=stable_checks,
    )
    asrf_losses = None
    asrf_summary = pd.DataFrame()
    asrf_sensitivity_summary = None
    method_comparison_summary = None

    if asrf_rho_grid:
        asrf_sensitivity_summary = estimate_ec_one_factor_sensitivity(
            ec_df,
            rho_grid=tuple(asrf_rho_grid),
            n_sim=n_sim,
            confidence_levels=confidence_levels,
            seed=seed,
            batch_size=batch_size,
            auto_stop=auto_stop,
            min_sim=min_sim,
            check_every=check_every,
            rel_tol=rel_tol,
            target_confidence=target_confidence,
            stable_checks=stable_checks,
        )
        _, asrf_basel_summary = estimate_ec_one_factor_monte_carlo(
            ec_df,
            n_sim=n_sim,
            confidence_levels=confidence_levels,
            rho=asrf_rho,
            rho_mode="basel",
            seed=seed + 999,
            batch_size=batch_size,
            auto_stop=auto_stop,
            min_sim=min_sim,
            check_every=check_every,
            rel_tol=rel_tol,
            target_confidence=target_confidence,
            stable_checks=stable_checks,
        )
        method_comparison_summary = _build_ec_method_comparison_summary(
            independent_summary=independent_summary,
            asrf_constant_summary=asrf_sensitivity_summary,
            asrf_basel_summary=asrf_basel_summary,
        )
        if not asrf_sensitivity_summary.empty:
            base_rho = float(tuple(asrf_rho_grid)[0])
            asrf_summary = asrf_sensitivity_summary[
                np.isclose(asrf_sensitivity_summary["rho"], base_rho)
            ].copy()
    else:
        asrf_losses, asrf_summary = estimate_ec_one_factor_monte_carlo(
            ec_df,
            n_sim=n_sim,
            confidence_levels=confidence_levels,
            rho=asrf_rho,
            rho_mode=asrf_rho_mode,
            seed=seed,
            batch_size=batch_size,
            auto_stop=auto_stop,
            min_sim=min_sim,
            check_every=check_every,
            rel_tol=rel_tol,
            target_confidence=target_confidence,
            stable_checks=stable_checks,
        )
        method_comparison_summary = _build_ec_method_comparison_summary(
            independent_summary=independent_summary,
            asrf_constant_summary=asrf_summary if str(asrf_rho_mode).lower() == "constant" else pd.DataFrame(),
            asrf_basel_summary=asrf_summary if str(asrf_rho_mode).lower() == "basel" else pd.DataFrame(),
        )

    realized_summary, grade_summary = realized_loss_backtest(ec_df)

    figure_dir = Path(output_dir) / "figures"
    try:
        independent_visualization_path = create_ec_visualizations(
            simulated_losses=independent_losses,
            monte_carlo_summary=independent_summary,
            portfolio_EL=portfolio_el_summary["portfolio_EL"],
            output_dir=figure_dir,
            file_name="portfolio_loss_distribution_var_ec_independent.png",
        )
        if asrf_losses is not None and len(asrf_losses) > 0 and not asrf_summary.empty:
            asrf_visualization_path = create_ec_visualizations(
                simulated_losses=asrf_losses,
                monte_carlo_summary=asrf_summary,
                portfolio_EL=portfolio_el_summary["portfolio_EL"],
                output_dir=figure_dir,
                file_name="portfolio_loss_distribution_var_ec_asrf.png",
            )
        else:
            asrf_visualization_path = None
        if method_comparison_summary is not None and not method_comparison_summary.empty:
            comparison_visual_paths = create_ec_method_comparison_visualizations(
                method_comparison_summary=method_comparison_summary,
                output_dir=figure_dir,
            )
    except ImportError:
        independent_visualization_path = None
        asrf_visualization_path = None
        comparison_visual_paths = {}

    save_ec_outputs(
        ec_df=ec_df,
        portfolio_el_summary=portfolio_el_summary,
        independent_summary=independent_summary,
        asrf_summary=asrf_summary,
        asrf_sensitivity_summary=asrf_sensitivity_summary,
        method_comparison_summary=method_comparison_summary,
        realized_summary=realized_summary,
        grade_summary=grade_summary,
        output_dir=output_dir,
    )

    return {
        "ec_df": ec_df,
        "portfolio_el_summary": portfolio_el_summary,
        "independent_monte_carlo_summary": independent_summary,
        "asrf_monte_carlo_summary": asrf_summary,
        "asrf_sensitivity_summary": asrf_sensitivity_summary,
        "method_comparison_summary": method_comparison_summary,
        "independent_simulated_losses": independent_losses,
        "asrf_simulated_losses": asrf_losses,
        "realized_summary": realized_summary,
        "grade_summary": grade_summary,
        "independent_visualization_path": independent_visualization_path,
        "asrf_visualization_path": asrf_visualization_path,
        "comparison_visual_paths": comparison_visual_paths,
    }


def _first_existing_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _extract_var(monte_carlo_summary, confidence_level):
    mask = np.isclose(monte_carlo_summary["confidence_level"].to_numpy(dtype=float), confidence_level)
    if not np.any(mask):
        return None
    return float(monte_carlo_summary.loc[mask, "VaR"].iloc[0])


def _validate_mc_params(n_sim, batch_size, min_sim, check_every, rel_tol, stable_checks):
    if int(n_sim) <= 0:
        raise ValueError("n_sim must be positive.")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive.")
    if int(min_sim) <= 0:
        raise ValueError("min_sim must be positive.")
    if int(check_every) <= 0:
        raise ValueError("check_every must be positive.")
    if int(stable_checks) <= 0:
        raise ValueError("stable_checks must be positive.")
    if float(rel_tol) < 0:
        raise ValueError("rel_tol must be non-negative.")


def _norm_ppf(probabilities):
    probs = np.asarray(probabilities, dtype=float)
    try:
        from scipy.stats import norm

        return norm.ppf(probs)
    except Exception:
        inv = np.vectorize(NormalDist().inv_cdf)
        return inv(probs)


def _norm_cdf(values):
    x = np.asarray(values, dtype=float)
    try:
        from scipy.stats import norm

        return norm.cdf(x)
    except Exception:
        cdf = np.vectorize(NormalDist().cdf)
        return cdf(x)


def build_basel_style_rho(pd_12m, decay=50.0, rho_low=0.12, rho_high=0.24):
    """
    Basel corporate-style rho(PD):
    rho_i = rho_low * w_i + rho_high * (1 - w_i),
    w_i = (1 - exp(-decay * PD_i)) / (1 - exp(-decay))
    """
    pd_vec = np.asarray(pd_12m, dtype=float)
    eps = 1e-12
    pd_vec = np.clip(pd_vec, eps, 1 - eps)
    denom = 1.0 - np.exp(-float(decay))
    if abs(denom) < 1e-12:
        raise ValueError("Invalid Basel rho decay parameter; denominator is too close to zero.")
    weight = (1.0 - np.exp(-float(decay) * pd_vec)) / denom
    rho_vec = float(rho_low) * weight + float(rho_high) * (1.0 - weight)
    return np.clip(rho_vec, 1e-10, 1 - 1e-10)


def _build_asrf_rho_vector(pd_12m, rho, rho_mode):
    mode = str(rho_mode).strip().lower()
    if mode == "constant":
        rho_scalar = float(rho)
        if not (0.0 <= rho_scalar < 1.0):
            raise ValueError("rho must be in [0, 1) for constant rho mode.")
        return np.full(len(pd_12m), rho_scalar, dtype=float)
    if mode == "basel":
        return build_basel_style_rho(pd_12m)
    raise ValueError("rho_mode must be either 'constant' or 'basel'.")


def _build_ec_method_comparison_summary(
    independent_summary,
    asrf_constant_summary,
    asrf_basel_summary,
):
    frames = []
    if independent_summary is not None and not independent_summary.empty:
        indep = independent_summary.copy()
        indep["rho_spec"] = "independent"
        frames.append(indep)
    if asrf_constant_summary is not None and not asrf_constant_summary.empty:
        const = asrf_constant_summary.copy()
        if "rho" in const.columns:
            const["rho_spec"] = const["rho"].map(lambda x: f"rho={x:.2f}")
        else:
            const["rho_spec"] = "rho=constant"
        frames.append(const)
    if asrf_basel_summary is not None and not asrf_basel_summary.empty:
        basel = asrf_basel_summary.copy()
        basel["rho_spec"] = "rho=basel"
        frames.append(basel)
    if not frames:
        return pd.DataFrame()

    summary = pd.concat(frames, ignore_index=True, sort=False)
    key_cols = ["confidence_level", "EC"]
    indep_ref = (
        summary[summary["method"].eq("independent_mc")][key_cols]
        .drop_duplicates(subset=["confidence_level"])
        .rename(columns={"EC": "ec_independent_ref"})
    )
    summary = summary.merge(indep_ref, on="confidence_level", how="left")
    summary["delta_ec_vs_independent"] = summary["EC"] - summary["ec_independent_ref"]
    return summary


def _compact_mc_summary_for_output(df):
    if df is None or df.empty:
        return df
    drop_columns = [
        "auto_stop",
        "converged",
        "n_sim_target",
        "min_sim",
        "check_every",
        "rel_tol",
        "target_confidence",
        "stable_checks",
    ]
    keep = [c for c in df.columns if c not in drop_columns]
    return df[keep].copy()


def create_ec_method_comparison_visualizations(method_comparison_summary, output_dir="outputs/figures"):
    """
    Visualize EC comparison across methods/rho settings and confidence levels.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise ImportError(
            "matplotlib is required to create EC comparison visualizations."
        ) from exc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = method_comparison_summary.copy()
    if df.empty:
        return {}

    df["confidence_level"] = pd.to_numeric(df["confidence_level"], errors="coerce")
    df = df.dropna(subset=["confidence_level", "EC"])
    if df.empty:
        return {}

    confidence_order = [0.99, 0.999, 0.9999]
    available_conf = [c for c in confidence_order if np.isclose(df["confidence_level"], c).any()]
    if not available_conf:
        available_conf = sorted(df["confidence_level"].dropna().unique().tolist())
    conf_labels = {0.99: "99%", 0.999: "99.9%", 0.9999: "99.99%"}

    # Plot 1: EC by confidence level for each method/rho scenario
    fig1, ax1 = plt.subplots(figsize=(11, 6))
    scenarios = df["rho_spec"].fillna(df["method"]).astype(str).unique().tolist()
    for scenario in scenarios:
        sub = df[df["rho_spec"].fillna(df["method"]).astype(str).eq(scenario)].copy()
        sub = sub.sort_values("confidence_level")
        if sub.empty:
            continue
        x = [conf_labels.get(float(c), f"{float(c):.4f}") for c in sub["confidence_level"]]
        ax1.plot(x, sub["EC"], marker="o", linewidth=2, label=scenario)
    ax1.set_title("Economic Capital Across Confidence Levels")
    ax1.set_xlabel("Confidence Level")
    ax1.set_ylabel("EC")
    ax1.grid(alpha=0.25, linestyle="--")
    ax1.legend(ncol=2, fontsize=9)
    fig1.tight_layout()
    ec_curve_path = output_dir / "ec_method_comparison_curves.png"
    fig1.savefig(ec_curve_path, dpi=150)
    plt.close(fig1)

    # Plot 2: EC delta vs independent by confidence level
    if "delta_ec_vs_independent" in df.columns:
        fig2, ax2 = plt.subplots(figsize=(11, 6))
        for scenario in scenarios:
            sub = df[df["rho_spec"].fillna(df["method"]).astype(str).eq(scenario)].copy()
            sub = sub.sort_values("confidence_level")
            if sub.empty:
                continue
            x = [conf_labels.get(float(c), f"{float(c):.4f}") for c in sub["confidence_level"]]
            ax2.plot(x, sub["delta_ec_vs_independent"], marker="o", linewidth=2, label=scenario)
        ax2.axhline(0, color="black", linewidth=1, linestyle=":")
        ax2.set_title("EC Delta vs Independent MC")
        ax2.set_xlabel("Confidence Level")
        ax2.set_ylabel("Delta EC")
        ax2.grid(alpha=0.25, linestyle="--")
        ax2.legend(ncol=2, fontsize=9)
        fig2.tight_layout()
        delta_curve_path = output_dir / "ec_delta_vs_independent_curves.png"
        fig2.savefig(delta_curve_path, dpi=150)
        plt.close(fig2)
    else:
        delta_curve_path = None

    paths = {
        "ec_method_comparison_curves": str(ec_curve_path),
    }
    if delta_curve_path is not None:
        paths["ec_delta_vs_independent_curves"] = str(delta_curve_path)
    return paths
