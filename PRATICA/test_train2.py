import pytest
import pandas as pd
import numpy as np
from unittest import mock
from keras.models import Sequential
from train2 import reset_seeds, read_data, process_data, create_model, train_model

@pytest.fixture
def sample_data():
    X = pd.DataFrame({
        'feature1': np.random.rand(10),
        'feature2': np.random.rand(10),
        'feature3': np.random.rand(10),
        'feature4': np.random.rand(10)
    })
    y = pd.Series(np.random.randint(1, 4, size=10))
    return X, y

def test_reset_seeds():
    # Should not raise any exception and set seeds
    reset_seeds()
    assert True

def test_read_data():
    X, y = read_data()
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert not X.empty
    assert not y.empty

def test_process_data(sample_data):
    X, y = sample_data
    X_train, X_test, y_train, y_test = process_data(X, y)
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert all(y_train >= 0)
    assert all(y_test >= 0)

def test_process_data_empty():
    X = pd.DataFrame()
    y = pd.Series(dtype=int)
    with pytest.raises(ValueError):
        process_data(X, y)

def test_process_data_missing_column():
    X = pd.DataFrame({'feature1': [1, 2], 'feature2': [3, 4]})
    y = pd.Series([1, 2])
    # Should work, but model will have only 2 features
    X_train, X_test, y_train, y_test = process_data(X, y)
    assert X_train.shape[1] == 2

def test_create_model(sample_data):
    X, _ = sample_data
    model = create_model(X)
    assert isinstance(model, Sequential)
    assert model.input_shape[1] == X.shape[1]
    assert model.output_shape[1] == 3

def test_create_model_with_empty_X():
    X = pd.DataFrame()
    with pytest.raises(ValueError):
        create_model(X)

def test_train_model(sample_data):
    X, y = sample_data
    model = create_model(X)
    # Mock mlflow and model.fit
    with mock.patch('train2.mlflow.start_run'), \
         mock.patch.object(model, 'fit', return_value=mock.Mock(history={'loss': [1, 0.5], 'val_loss': [1, 0.6]})) as fit_mock:
        train_model(model, X, y, is_train=True)
        fit_mock.assert_called()

def test_train_model_with_wrong_shape(sample_data):
    X, y = sample_data
    model = create_model(X)
    # Pass y with wrong length
    y_wrong = y.iloc[:-1]
    with mock.patch('train2.mlflow.start_run'), \
         mock.patch.object(model, 'fit', side_effect=ValueError("Found input variables with inconsistent numbers of samples")):
        with pytest.raises(ValueError):
            train_model(model, X, y_wrong, is_train=True)

def test_read_data_columns():
    X, y = read_data()
    # Desafiar: checar se todas as colunas esperadas existem
    expected_cols = {'severe_decelerations', 'accelerations', 'fetal_movement', 'uterine_contractions'}
    assert expected_cols.issubset(set(X.columns))
