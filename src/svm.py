import os
import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)

# ensure dirs
os.makedirs("models", exist_ok=True)
os.makedirs("results/plots", exist_ok=True)

# connect to MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("MLOps-Assignment-1")

def main():
    X, y = datasets.load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = SVC(C=1.0, kernel="rbf", probability=True, random_state=42)

    with mlflow.start_run(run_name="SVM"):
        mlflow.log_param("C", 1.0)
        mlflow.log_param("kernel", "rbf")

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

        # save model
        model_path = "models/svm_model.pkl"
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)
        mlflow.sklearn.log_model(model, artifact_path="model")

        # confusion matrix plot
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Blues")
        plt.title("SVM - Confusion Matrix")
        plt.tight_layout()
        plot_path = "results/plots/svm_cm.png"
        plt.savefig(plot_path)
        plt.close()
        mlflow.log_artifact(plot_path)

        print(f"SVM → acc={acc:.3f}, prec={prec:.3f}, rec={rec:.3f}, f1={f1:.3f}")

if __name__ == "__main__":
    main()
