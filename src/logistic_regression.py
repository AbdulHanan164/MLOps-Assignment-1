import os
import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)

os.makedirs("models", exist_ok=True)
os.makedirs("results/plots", exist_ok=True)

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("MLOps-Assignment-1")

def main():
    X, y = datasets.load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=200, random_state=42)

    with mlflow.start_run(run_name="LogisticRegression"):
        mlflow.log_param("max_iter", 200)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro")
        rec = recall_score(y_test, y_pred, average="macro")
        f1 = f1_score(y_test, y_pred, average="macro")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1", f1)

        model_path = "models/logistic_regression_model.pkl"
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)
        mlflow.sklearn.log_model(model, artifact_path="model")

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Oranges")
        plt.title("Logistic Regression - Confusion Matrix")
        plt.tight_layout()
        plot_path = "results/plots/logistic_regression_cm.png"
        plt.savefig(plot_path)
        plt.close()
        mlflow.log_artifact(plot_path)

        print(f"LogisticRegression → acc={acc:.3f}, prec={prec:.3f}, rec={rec:.3f}, f1={f1:.3f}")

if __name__ == "__main__":
    main()
