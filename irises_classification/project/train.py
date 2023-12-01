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

    train_data = train_data.values
    X_train = train_data[:, :-1]
    y_train = train_data[:, -1]

    return X_train, y_train


def train(X_train, y_train):
    log_reg = LogisticRegression(
        multi_class='multinomial', max_iter=500, solver='lbfgs', random_state=42
    ).fit(X_train, y_train)

    return log_reg


def save_model(model, model_path):
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    X_train, y_train = get_data(cfg.train.data_path)
    model = train(X_train, y_train)
    save_model(model, cfg.train.model_path)


if __name__ == '__main__':
    main()
