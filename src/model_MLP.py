from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_MLP_PARAMS = {
    "hidden_layer_sizes": (64,),
    "activation": "relu",
    "solver": "adam",
    "alpha": 0.0001,
    "learning_rate_init": 0.001,
    "max_iter": 120,
    "early_stopping": True,
    "validation_fraction": 0.1,
    "n_iter_no_change": 10,
    "random_state": 42,
}


def get_default_mlp_params():
    return DEFAULT_MLP_PARAMS.copy()


def build_mlp_model(params=None):
    model_params = get_default_mlp_params()
    if params:
        model_params.update(params)
    if isinstance(model_params.get("hidden_layer_sizes"), list):
        model_params["hidden_layer_sizes"] = tuple(model_params["hidden_layer_sizes"])

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", MLPClassifier(**model_params)),
        ]
    )
