import pickle

import hydra
import pandas as pd
from dvc.api import DVCFileSystem
from omegaconf import DictConfig
from sklearn.linear_model import LogisticRegression


def get_data(data_path):
    fs = DVCFileSystem("./")
    fs.get(data_path, data_path)
    fs.get(data_path, data_path)

    train_data = pd.read_csv(data_path)

    X_train = train_data[train_data.columns[train_data.columns != train_data.columns[-1]]]
    y_train = train_data[[train_data.columns[-1]]]

    return X_train, y_train


def train(X_train, y_train, params):
    log_reg = LogisticRegression(**params).fit(X_train, y_train)

    return log_reg


def save_model(model, model_path):
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    X_train, y_train = get_data(cfg.train_params.data_path)
    model = train(X_train, y_train, cfg.model_params)
    save_model(model, cfg.train_params.model_path)


if __name__ == '__main__':
    main()
