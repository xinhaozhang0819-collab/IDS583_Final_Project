from xgboost import XGBClassifier


DEFAULT_XGB_PARAMS = {
    "n_estimators": 120,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
}


def get_default_xgb_params():
    return DEFAULT_XGB_PARAMS.copy()


def build_xgboost_model(params=None):
    model_params = get_default_xgb_params()
    if params:
        model_params.update(params)

    return XGBClassifier(**model_params)
