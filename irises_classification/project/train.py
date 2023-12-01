import argparse
import pickle

import pandas as pd
from dvc.api import DVCFileSystem
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data_path', type=str, default='./irises_classification/dataset/train.csv'
    )
    parser.add_argument(
        '--model_path', type=str, default='./irises_classification/models/model.pkl'
    )
    args = parser.parse_args()

    X_train, y_train = get_data(args.data_path)
    model = train(X_train, y_train)
    save_model(model, args.model_path)
