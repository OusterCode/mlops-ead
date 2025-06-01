import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
import mlflow

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def reset_seeds() -> None:
    """Reseta as seeds para reprodutibilidade."""
    os.environ['PYTHONHASHSEED'] = str(42)
    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)


def read_data() -> tuple[pd.DataFrame, pd.Series]:
    """Função para ler o dataset"""
    url = 'raw.githubusercontent.com'
    username = 'renansantosmendes'
    repository = 'lectures-cdas-2023'
    file_name = 'fetal_health_reduced.csv'
    full_url = f'https://{url}/{username}/{repository}/master/{file_name}'
    try:
        data = pd.read_csv(full_url)
    except Exception as e:
        raise RuntimeError(f"Erro ao ler o dataset: {e}")
    X = data.drop(["fetal_health"], axis=1)
    y = data["fetal_health"]
    return X, y

def process_data(X: pd.DataFrame, y: pd.Series):
    """Padroniza e divide os dados."""
    columns_names = list(X.columns)
    scaler = StandardScaler()
    X_df = scaler.fit_transform(X)
    X_df = pd.DataFrame(X_df, columns=columns_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.3, random_state=42
    )

    y_train = y_train - 1
    y_test = y_test - 1

    return X_train, X_test, y_train, y_test


def create_model(X_train: pd.DataFrame) -> Sequential:
    """Cria o modelo Keras."""
    if X_train.empty:
        raise ValueError("X_train está vazio. Não é possível criar o modelo.")
    reset_seeds()
    model = Sequential([
        InputLayer(shape=(X_train.shape[1], )),
        Dense(units=10, activation='relu'),
        Dense(units=10, activation='relu'),
        Dense(units=3, activation='softmax')
    ])

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    return model


def config_mlflow():
    """Configura o mlflow."""
    os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv('MLFLOW_TRACKING_USERNAME', 'oustercode')
    os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('MLFLOW_TRACKING_PASSWORD', 'b933b0cdbe9cf414db8ed938c5488a02d8d1f25d')
    mlflow.set_tracking_uri('https://dagshub.com/oustercode/puc-mlops-class.mlflow')

    mlflow.tensorflow.autolog(
        log_models=True,
        log_input_examples=True,
        log_model_signatures=True
    )


def train_model(model: Sequential, X_train, y_train, is_train=True):
    """Executa o treino do modelo."""
    with mlflow.start_run(run_name='experiment_mlops_ead'):
        model.fit(
            X_train,
            y_train,
            epochs=50,
            validation_split=0.2,
            verbose=3
        )

if __name__ == "__main__":
    reset_seeds()
    X, y = read_data()
    X_train, X_test, y_train, y_test = process_data(X, y)
    model = create_model(X_train)
    config_mlflow()
    train_model(model, X_train, y_train)
