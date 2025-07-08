import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
import joblib
from prometheus_client import Summary, Gauge, start_http_server
from pathlib import Path

# PROMETHEUS METRICS
TRAINING_TIME = Summary('model_training_seconds', 'Time spent training the model')
MODEL_ACCURACY = Gauge('model_accuracy', 'Accuracy of the trained model')
MODEL_AUC = Gauge('model_auc', 'ROC AUC of the trained model')
MODEL_PRECISION = Gauge('model_precision', 'Precision of the trained model')
MODEL_RECALL = Gauge('model_recall', 'Recall of the trained model')
MODEL_F1 = Gauge('model_f1', 'F1 Score of the trained model')


@TRAINING_TIME.time()
def train():
    print("[INFO] Reading features...")
    df = pd.read_parquet("data/features/v1_features.parquet")

    # Define features and target
    if "user_id" in df.columns:
        df = df.drop(columns=["user_id"])  # drop user_id if exists

    features = df.drop(columns=["default"])
    target = df["default"]

    print("[INFO] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
        )

    print("[INFO] Training model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    print("[INFO] Evaluating model...")
    preds = model.predict(X_test)
    probas = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probas)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    report = classification_report(y_test, preds, digits=4)
    cm = confusion_matrix(y_test, preds)

    print(f"[METRIC] Accuracy: {acc:.4f}")
    print(f"[METRIC] ROC AUC: {auc:.4f}")
    print(f"[METRIC] Precision: {precision:.4f}")
    print(f"[METRIC] Recall: {recall:.4f}")
    print(f"[METRIC] F1 Score: {f1:.4f}")
    print(f"\n[INFO] Classification Report:\n{report}")
    print(f"\n[INFO] Confusion Matrix:\n{cm}")

    # Save metrics to Prometheus
    MODEL_ACCURACY.set(acc)
    MODEL_AUC.set(auc)
    MODEL_PRECISION.set(precision)
    MODEL_RECALL.set(recall)
    MODEL_F1.set(f1)

    print("[INFO] Saving model and reports...")
    Path("models/").mkdir(parents=True, exist_ok=True)
    joblib.dump(model, "models/model_v1.pkl")

    # Save classification report and confusion matrix to file
    with open("models/evaluation_report.txt", "w") as f:
        f.write("CLASSIFICATION REPORT\n")
        f.write(report)
        f.write("\nCONFUSION MATRIX\n")
        f.write(str(cm))

    return model


if __name__ == "__main__":
    print("[INFO] Starting Prometheus exporter on :8000...")
    start_http_server(8000)  # Prometheus scrapes metrics from here
    train()