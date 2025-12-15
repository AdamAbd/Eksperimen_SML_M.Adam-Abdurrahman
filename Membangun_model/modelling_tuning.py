import os
import json
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression


def load_data():
    data_path = os.path.join("namadataset_preprocessing", "titanic_preprocessing.csv")
    df = pd.read_csv(data_path)

    if "Survived" not in df.columns:
        raise ValueError("Kolom target 'Survived' tidak ditemukan di dataset preprocessing.")

    X = df.drop(columns=["Survived"])
    y = df["Survived"]
    return X, y


def eval_metrics(y_true, y_pred, y_proba=None):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba) if y_proba is not None else None
    cm = confusion_matrix(y_true, y_pred)
    return acc, f1, auc, cm


def main():
    # 1) Load dataset preprocessing
    X, y = load_data()

    # 2) Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3) Setup MLflow experiment
    mlflow.set_experiment("Titanic-MLflow-Manual-Tuning")

    # 4) Model + hyperparameter tuning
    base_model = LogisticRegression(max_iter=3000)

    param_grid = {
        "C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "solver": ["lbfgs", "liblinear"],
    }

    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1,
        verbose=0,
    )

    # 5) Manual MLflow logging
    with mlflow.start_run(run_name="logreg_gridsearch_manual"):
        # Log info dataset
        mlflow.log_param("dataset_path", "namadataset_preprocessing/titanic_preprocessing.csv")
        mlflow.log_param("n_samples", int(len(X)))
        mlflow.log_param("n_features", int(X.shape[1]))
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("scoring", "f1")

        # Fit tuning
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        best_params = grid.best_params_
        best_cv_score = grid.best_score_

        # Log best params + best CV score
        for k, v in best_params.items():
            mlflow.log_param(f"best_{k}", v)
        mlflow.log_metric("best_cv_f1", float(best_cv_score))

        # Evaluate on test set
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None

        acc, f1, auc, cm = eval_metrics(y_test, y_pred, y_proba)

        mlflow.log_metric("test_accuracy", float(acc))
        mlflow.log_metric("test_f1", float(f1))
        if auc is not None:
            mlflow.log_metric("test_auc", float(auc))

        # Log confusion matrix as artifact (json)
        cm_dict = {
            "confusion_matrix": cm.tolist(),
            "labels": ["0 (not survived)", "1 (survived)"]
        }
        os.makedirs("artifacts", exist_ok=True)
        cm_path = os.path.join("artifacts", "confusion_matrix.json")
        with open(cm_path, "w") as f:
            json.dump(cm_dict, f, indent=2)
        mlflow.log_artifact(cm_path)

        # Log model as artifact
        mlflow.sklearn.log_model(best_model, artifact_path="model")

        print("✅ Tuning selesai (manual logging)")
        print("Best Params:", best_params)
        print("Best CV F1 :", best_cv_score)
        print("Test Acc   :", acc)
        print("Test F1    :", f1)
        print("Test AUC   :", auc)


if __name__ == "__main__":
    main()
