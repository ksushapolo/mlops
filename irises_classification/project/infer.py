import pickle

import git
import hydra
import mlflow
import pandas as pd
from dvc.api import DVCFileSystem
from mlflow.models import infer_signature
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, precision_score, recall_score


def get_data(data_path):
    fs = DVCFileSystem("./")
    fs.get(data_path, data_path)
    fs.get(data_path, data_path)

    val_data = pd.read_csv(data_path)

    X_val = val_data[val_data.columns[val_data.columns != val_data.columns[-1]]]
    y_val = val_data[[val_data.columns[-1]]]

    return X_val, y_val


def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    return model


def save_prediction(y_pred, prediction_path):
    y_pred_df = pd.DataFrame(y_pred, columns=["target"])
    y_pred_df.to_csv(prediction_path, index=False)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    X_val, y_val = get_data(cfg.infer_params.data_path)
    model = load_model(cfg.infer_params.model_path)

    y_pred = model.predict(X_val)

    metrics = {}

    metrics["accuracy"] = accuracy_score(y_val, y_pred)
    metrics["precision_score"] = precision_score(y_val, y_pred, average="macro")
    metrics["recall_score"] = recall_score(y_val, y_pred, average="macro")

    print(metrics)

    save_prediction(y_pred, cfg.infer_params.prediction_path)

    mlflow.set_tracking_uri(uri=cfg.server_params.uri)

    mlflow.set_experiment(cfg.server_params.exp_name)

    with mlflow.start_run():
        mlflow.log_params(cfg.model_params)

        mlflow.log_metrics(metrics)

        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha

        mlflow.set_tag("git_commit_id", sha)

        signature = infer_signature(X_val, y_val)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="iris_model",
            signature=signature,
            input_example=X_val,
            registered_model_name="log_reg",
        )


if __name__ == '__main__':
    main()
