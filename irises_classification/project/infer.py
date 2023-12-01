import pickle

import hydra
import pandas as pd
from dvc.api import DVCFileSystem
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score


def get_data(data_path):
    fs = DVCFileSystem("./")
    fs.get(data_path, data_path)
    fs.get(data_path, data_path)

    val_data = pd.read_csv(data_path)

    val_data = val_data.values
    X_val = val_data[:, :-1]
    y_val = val_data[:, -1]

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
    X_val, y_val = get_data(cfg.infer.data_path)
    model = load_model(cfg.infer.model_path)

    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print("Точность: {}".format(accuracy))

    save_prediction(y_pred, cfg.infer.prediction_path)


if __name__ == '__main__':
    main()
