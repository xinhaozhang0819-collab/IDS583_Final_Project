from sklearn.ensemble import RandomForestClassifier


DEFAULT_RF_PARAMS = {
    "n_estimators": 120,
    "max_depth": 12,
    "min_samples_split": 100,
    "min_samples_leaf": 25,
    "max_features": "sqrt",
    "n_jobs": -1,
    "random_state": 42,
}


def get_default_rf_params():
    return DEFAULT_RF_PARAMS.copy()


def build_random_forest_model(params=None):
    model_params = get_default_rf_params()
    if params:
        model_params.update(params)

    return RandomForestClassifier(**model_params)
