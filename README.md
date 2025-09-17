# MLOps Assignment 1 — Report

## 1. Problem statement & dataset description

### Problem statement

Build an MLOps workflow to train, evaluate and register machine learning models for a multi-class classification problem. The workflow must:

* Train multiple candidate models.
* Log parameters, metrics and artifacts to MLflow.
* Compare models and pick the best model using a defined selection metric.
* Register the best model in the MLflow Model Registry for later deployment.

### Dataset

For this assignment we use the **Iris** dataset (scikit‑learn builtin):

* 150 samples, 3 classes (`setosa`, `versicolor`, `virginica`).
* 4 features: sepal length, sepal width, petal length, petal width.

*Note:* The code is written so the dataset can be swapped easily (e.g., Wine, Breast Cancer) if you change the `datasets.load_*` call.

---

## 2. Project structure

```
MLOPs-Assignment-1/
├─ data/
├─ notebooks/
├─ src/
│  ├─ svm.py
│  ├─ random_forest.py
│  ├─ logistic_regression.py
│  └─ register_best_model.py
├─ models/
├─ results/
│  └─ plots/
├─ mlruns/         # mlflow local artifacts (if used)
├─ mlflow.db       # sqlite backend (if used)
├─ requirements.txt
└─ README.md
```

---

## 3. Model selection & comparison

### Models trained

* **Logistic Regression** — `sklearn.linear_model.LogisticRegression` (param: `max_iter`)
* **Random Forest** — `sklearn.ensemble.RandomForestClassifier` (param: `n_estimators`)
* **Support Vector Machine (SVM)** — `sklearn.svm.SVC` (params: `C`, `kernel`)

All models are trained on the same train/test split (80/20, `random_state=42`) for a fair comparison.

### Logged items for each model

* **Parameters (hyperparameters)** — e.g., `n_estimators`, `C`, `kernel`, `max_iter`.
* **Metrics** — `accuracy`, `precision` (macro), `recall` (macro), `f1` (macro).
* **Artifacts** — saved model file (`.pkl`), confusion matrix image (`.png`).

### Selection criterion

The best model is selected **automatically** by choosing the run with the **highest F1 score** (macro). This is implemented in `src/register_best_model.py`.

---

## 4. MLflow logging (where to find things)

After running the training scripts and pointing them to your MLflow server, open the MLflow UI at:

```
http://127.0.0.1:5000
```

Inside MLflow UI:

* **Experiments** → open `MLOps-Assignment-1` → see all runs (one run per model).
* Click a run to view:

  * **Parameters** (left panel)
  * **Metrics** (left panel)
  * **Artifacts** (contains saved model and `results/plots/{model}_cm.png`)

### MLflow screenshots (placeholders)

Place screenshots in `results/screenshots/` and commit them with the README. Example placeholders are included below; replace them with your actual screenshots:

* `results/screenshots/mlflow_experiments.png` — Experiments page showing all runs.
* `results/screenshots/run_metrics.png` — Single-run detail showing parameters & metrics.
* `results/screenshots/artifacts_confusion_matrix.png` — Run artifacts showing confusion matrix image.

To embed the screenshots in this README (after you take them), add these markdown lines where needed:

```markdown
![MLflow experiments](results/screenshots/mlflow_experiments.png)
![Run metrics](results/screenshots/run_metrics.png)
![Confusion matrix artifact](results/screenshots/artifacts_confusion_matrix.png)
```

---

## 5. Model registration (what we did & where to see it)

We register the best model using the MLflow Model Registry. The script `src/register_best_model.py`:

* Connects to MLflow server at `http://127.0.0.1:5000`.
* Finds the best run in experiment `MLOps-Assignment-1` by sorting runs on `metrics.f1` (descending).
* Registers `runs:/{best_run_id}/model` under the registry name `Best_Iris_Model`.

After running the registration script, open MLflow UI → **Models** tab → select `Best_Iris_Model`.

### Model registration screenshots (placeholders)

Save these screenshots to `results/screenshots/` and replace the placeholders in README:

* `results/screenshots/model_registry_list.png` — Models tab listing the registered model.
* `results/screenshots/model_version_details.png` — Model version page showing source run and artifacts.

Insert with markdown like:

```markdown
![Model registry list](results/screenshots/model_registry_list.png)
![Model version details](results/screenshots/model_version_details.png)
```

---

## 6. How to run (step-by-step)

All commands assume Windows PowerShell and project root `MLOPs-Assignment-1`. Substitute paths for other shells.

### 6.1 Create & activate virtual environment

```powershell
python -m venv .venv
# PowerShell activate
.venv\Scripts\Activate.ps1
# (If execution policy blocks activation, open PyCharm interpreter or use cmd.exe and run .venv\Scripts\activate.bat)

# Upgrade pip & setuptools to avoid distutils issues
pip install --upgrade pip setuptools wheel
```

### 6.2 Install dependencies

```powershell
pip install -r requirements.txt
```

### 6.3 Start MLflow server (leave this terminal open)

```powershell
python -m mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000
```

> Keep the MLflow server running in **its own terminal window**. Everything below runs in a separate terminal where the venv is active.

### 6.4 Run training scripts (logs go to MLflow server)

```powershell
# In a second terminal (activate venv first)
.venv\Scripts\Activate.ps1
python src\svm.py
python src\random_forest.py
python src\logistic_regression.py
```

### 6.5 Register best model

```powershell
python src\register_best_model.py
```

### 6.6 Open MLflow UI

Open your browser to `http://127.0.0.1:5000` to view experiments, runs and the Model Registry.

---

## 7. How to load a registered model for inference

Example Python snippet to load model version 1 from the registry and run a prediction:

```python
import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
model = mlflow.pyfunc.load_model("models:/Best_Iris_Model/1")
print(model.predict([[5.1, 3.5, 1.4, 0.2]]))
```

Replace `1` with the appropriate version number shown in the Model Registry.

---
