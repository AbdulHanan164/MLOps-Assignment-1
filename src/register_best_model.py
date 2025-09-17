import mlflow
from mlflow.tracking import MlflowClient

# ✅ connect to MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5000")
EXPERIMENT_NAME = "MLOps-Assignment-1"
MODEL_NAME = "Best_Iris_Model"

def main():
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        print("❌ Experiment not found. Run your training scripts first.")
        return

    # get all runs
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.f1 DESC"],  # best by F1
        max_results=1
    )

    if not runs:
        print("❌ No runs found in experiment.")
        return

    best_run = runs[0]
    best_run_id = best_run.info.run_id
    best_f1 = best_run.data.metrics["f1"]
    best_model = best_run.data.params["model"] if "model" in best_run.data.params else "Unknown"

    print(f"✅ Best model: {best_model} (Run ID: {best_run_id}, F1={best_f1:.3f})")

    # Register model
    result = mlflow.register_model(
        model_uri=f"runs:/{best_run_id}/model",
        name=MODEL_NAME
    )

    print(f"📌 Model registered as '{MODEL_NAME}' (Version: {result.version})")

if __name__ == "__main__":
    main()
