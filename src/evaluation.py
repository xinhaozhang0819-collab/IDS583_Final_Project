from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp


def evaluate_model(y_true, y_pred_proba):
    auc = roc_auc_score(y_true, y_pred_proba)
    ks = ks_2samp(y_pred_proba[y_true == 1], y_pred_proba[y_true == 0]).statistic

    return {"AUC": auc, "KS": ks}
