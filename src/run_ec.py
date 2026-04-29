from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ec import run_ec_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "test_with_loss_metrics.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs_ec_test"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Economic Capital (EC) pipeline.")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Input CSV path (default: {DEFAULT_INPUT_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument("--n-sim", type=int, default=50000, help="Maximum Monte Carlo simulations.")
    parser.add_argument("--batch-size", type=int, default=200, help="Monte Carlo batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--confidence-levels",
        type=float,
        nargs="+",
        default=[0.99, 0.999, 0.9999],
        help="Confidence levels for VaR/EC (e.g. --confidence-levels 0.99 0.999 0.9999).",
    )
    parser.add_argument("--asrf-rho", type=float, default=0.10, help="ASRF one-factor asset correlation rho.")
    parser.add_argument(
        "--asrf-rho-mode",
        type=str,
        choices=["constant", "basel"],
        default="constant",
        help="ASRF rho mode: constant or basel.",
    )
    parser.add_argument(
        "--asrf-rho-grid",
        type=float,
        nargs="+",
        default=None,
        help="Optional ASRF constant-rho sensitivity grid (e.g. --asrf-rho-grid 0 0.05 0.10 0.20). "
        "When provided, Basel-style rho is also run and both are merged into one comparison CSV.",
    )
    parser.add_argument(
        "--auto-stop",
        action="store_true",
        help="Enable automatic convergence stopping.",
    )
    parser.add_argument("--min-sim", type=int, default=10000, help="Minimum sims before auto-stop check.")
    parser.add_argument("--check-every", type=int, default=5000, help="Check interval for auto-stop.")
    parser.add_argument("--rel-tol", type=float, default=0.01, help="Relative tolerance for VaR stabilization.")
    parser.add_argument("--target-confidence", type=float, default=0.99, help="Target confidence for convergence.")
    parser.add_argument("--stable-checks", type=int, default=2, help="Required consecutive stable checks.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path = args.input_path
    output_dir = args.output_dir

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    result = run_ec_pipeline(
        df=df,
        n_sim=args.n_sim,
        confidence_levels=tuple(args.confidence_levels),
        seed=args.seed,
        batch_size=args.batch_size,
        auto_stop=args.auto_stop,
        min_sim=args.min_sim,
        check_every=args.check_every,
        rel_tol=args.rel_tol,
        target_confidence=args.target_confidence,
        stable_checks=args.stable_checks,
        asrf_rho=args.asrf_rho,
        asrf_rho_mode=args.asrf_rho_mode,
        asrf_rho_grid=args.asrf_rho_grid,
        output_dir=str(output_dir),
    )

    portfolio_el = result["portfolio_el_summary"]["portfolio_EL"]
    indep_summary = result["independent_monte_carlo_summary"]
    asrf_summary = result["asrf_monte_carlo_summary"]
    indep_var99_rows = indep_summary[indep_summary["confidence_level"].round(6).eq(round(0.99, 6))]
    indep_ec99 = float(indep_var99_rows["EC"].iloc[0]) if not indep_var99_rows.empty else float("nan")

    print(f"EC pipeline completed. Input: {input_path}")
    print(f"Outputs written to: {output_dir}")
    print(f"Portfolio EL (12m): {portfolio_el:.6f}")
    print(f"EC@99% (Independent MC): {indep_ec99:.6f}")
    if {"converged", "n_sim", "n_sim_target"}.issubset(indep_summary.columns):
        indep_row = indep_summary.iloc[0]
        print(
            "Independent MC convergence: "
            f"{bool(indep_row['converged'])} "
            f"(n_sim_used={int(indep_row['n_sim'])}, n_sim_target={int(indep_row['n_sim_target'])})"
        )
    if args.asrf_rho_grid:
        print(f"ASRF comparison saved for constant grid {args.asrf_rho_grid} plus Basel-style rho.")
        print("See: ec_method_comparison_summary.csv")
    else:
        asrf_var99_rows = asrf_summary[asrf_summary["confidence_level"].round(6).eq(round(0.99, 6))]
        asrf_ec99 = float(asrf_var99_rows["EC"].iloc[0]) if not asrf_var99_rows.empty else float("nan")
        if args.asrf_rho_mode == "basel":
            print(f"EC@99% (ASRF Vasicek, Basel rho): {asrf_ec99:.6f}")
        else:
            print(f"EC@99% (ASRF Vasicek, rho={args.asrf_rho:.4f}): {asrf_ec99:.6f}")


if __name__ == "__main__":
    main()
