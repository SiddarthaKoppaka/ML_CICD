from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def get_model(model_type, params):
    if model_type == "logistic_regression":
        return LogisticRegression(**params)
    elif model_type == "random_forest":
        return RandomForestClassifier(**params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
