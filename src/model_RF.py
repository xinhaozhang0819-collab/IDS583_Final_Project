from sklearn.ensemble import RandomForestClassifier


def train_rf(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=50,
        min_samples_leaf=20,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def predict_pd_rf(model, X):
    return model.predict_proba(X)[:, 1]
