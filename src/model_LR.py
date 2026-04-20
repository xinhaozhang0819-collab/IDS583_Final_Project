from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_LOGISTIC_PARAMS = {
    "C": 1.0,
    "max_iter": 1000,
    "solver": "lbfgs",
    "class_weight": None,
}


def get_default_logistic_params():
    return DEFAULT_LOGISTIC_PARAMS.copy()


def build_logistic_model(params=None):
    model_params = get_default_logistic_params()
    if params:
        model_params.update(params)

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(**model_params)),
        ]
    )
