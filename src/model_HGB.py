from sklearn.ensemble import HistGradientBoostingClassifier


DEFAULT_HGB_PARAMS = {
    "learning_rate": 0.05,
    "max_iter": 150,
    "max_depth": 6,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 50,
    "l2_regularization": 0.0,
    "early_stopping": False,
}


def get_default_hgb_params():
    return DEFAULT_HGB_PARAMS.copy()


def build_hist_gradient_boosting_model(params=None):
    model_params = get_default_hgb_params()
    if params:
        model_params.update(params)

    return HistGradientBoostingClassifier(**model_params)
