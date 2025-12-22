import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()

    # Load dataset
    df = pd.read_csv(args.data_path)

    X = df.drop("popular", axis=1)
    y = df["popular"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run():

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42
        )

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        # Manual logging
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 10)
        mlflow.log_metric("accuracy", acc)

        mlflow.sklearn.log_model(model, "model")

    print("Training selesai dan berhasil dicatat MLflow.")
