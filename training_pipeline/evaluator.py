from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
import json


def evaluate_model(model, X_test, y_test, metrics_path):
    preds = model.predict(X_test)
    probas = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, preds)
    roc_auc = roc_auc_score(y_test, probas)
    report = classification_report(y_test, preds, output_dict=True)
    cm = confusion_matrix(y_test, preds)

    metrics = {
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics
