import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression

def main():
    # 1) Load data hasil preprocessing (WAJIB bukan raw)
    data_path = os.path.join("namadataset_preprocessing", "titanic_preprocessing.csv")
    df = pd.read_csv(data_path)

    # 2) Split fitur & target
    if "Survived" not in df.columns:
        raise ValueError("Kolom target 'Survived' tidak ditemukan di dataset preprocessing.")
    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3) Aktifkan MLflow autolog (syarat Basic)
    mlflow.set_experiment("Titanic-MLflow-Autolog")
    mlflow.sklearn.autolog(log_models=True)

    # 4) Train model
    # LogisticRegression sederhana dan cepat untuk lolos Basic
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # 5) Evaluasi tambahan (opsional, tapi membantu)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # ROC AUC butuh probabilitas
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
    else:
        auc = None

    print("Accuracy:", acc)
    print("F1:", f1)
    print("AUC:", auc)

    # Autolog biasanya sudah mencatat metrics dari training,
    # tapi metrics evaluasi ini bisa kamu log manual juga (aman walau Basic).
    with mlflow.start_run(run_name="logreg_autolog_eval"):
        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_f1", f1)
        if auc is not None:
            mlflow.log_metric("test_auc", auc)

if __name__ == "__main__":
    main()
