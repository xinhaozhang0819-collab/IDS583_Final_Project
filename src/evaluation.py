import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve


def compute_ks(y_true, y_pred_proba):
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    return ks_2samp(y_pred_proba[y_true == 1], y_pred_proba[y_true == 0]).statistic


def summarize_calibration(prob_true, prob_pred):
    if len(prob_true) == 0:
        return "Calibration unavailable."

    gap = np.asarray(prob_true) - np.asarray(prob_pred)
    mean_abs_gap = float(np.mean(np.abs(gap)))
    high_risk_gap = float(gap[-1])

    if mean_abs_gap <= 0.02 and abs(high_risk_gap) <= 0.03:
        return "Well aligned across calibration bins."
    if high_risk_gap > 0.03:
        return "Underestimates risk in the highest-PD bins."
    if high_risk_gap < -0.03:
        return "Overestimates risk in the highest-PD bins."
    if mean_abs_gap <= 0.04:
        return "Mostly aligned with mild bin-level drift."
    return "Visible calibration drift; inspect the curve before finalizing."


def build_calibration_table(y_true, y_pred_proba, n_bins=10):
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)

    clipped_pred = np.clip(y_pred_proba, 0, 1 - 1e-12)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(clipped_pred, bin_edges[1:-1], right=False)

    rows = []
    for bin_index in range(n_bins):
        mask = bin_ids == bin_index
        if not np.any(mask):
            continue
        rows.append(
            {
                "bin_index": int(bin_index),
                "bin_left": float(bin_edges[bin_index]),
                "bin_right": float(bin_edges[bin_index + 1]),
                "prob_true": float(y_true[mask].mean()),
                "prob_pred": float(y_pred_proba[mask].mean()),
                "bin_count": int(mask.sum()),
            }
        )

    return rows


def evaluate_model(y_true, y_pred_proba, n_bins=10):
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)

    calibration_rows = build_calibration_table(y_true, y_pred_proba, n_bins=n_bins)
    prob_true = np.asarray([row["prob_true"] for row in calibration_rows])
    prob_pred = np.asarray([row["prob_pred"] for row in calibration_rows])
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)

    return {
        "AUC": float(roc_auc_score(y_true, y_pred_proba)),
        "KS": float(compute_ks(y_true, y_pred_proba)),
        "Brier": float(brier_score_loss(y_true, y_pred_proba)),
        "roc_curve": {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds,
        },
        "calibration_curve": {
            "prob_true": prob_true,
            "prob_pred": prob_pred,
            "bin_count": np.asarray([row["bin_count"] for row in calibration_rows]),
            "bin_left": np.asarray([row["bin_left"] for row in calibration_rows]),
            "bin_right": np.asarray([row["bin_right"] for row in calibration_rows]),
        },
        "calibration_summary": summarize_calibration(prob_true, prob_pred),
    }
