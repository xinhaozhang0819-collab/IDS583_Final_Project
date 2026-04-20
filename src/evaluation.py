import numpy as np
from scipy.stats import ks_2samp
from sklearn.calibration import calibration_curve
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


def evaluate_model(y_true, y_pred_proba, n_bins=10):
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)

    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=n_bins)
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
        },
        "calibration_summary": summarize_calibration(prob_true, prob_pred),
    }
