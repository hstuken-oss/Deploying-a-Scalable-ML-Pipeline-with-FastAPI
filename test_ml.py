import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model, compute_model_metrics, inference


def test_train_model():
    """
    Test that the train_model returns a Random Forest Classifier that 
    has been fitted on the training data.
    """
    X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y_train = np.array([0, 1, 0, 1])
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)


def test_compute_model_metrics():
    """
    Test that the compute_model_metrics returns precision, recall, and fbeta
     as floats with values betweeen 0 and 1.
    """
    y = np.array([1, 0, 1, 0, 1])
    preds = np.array([1, 0, 1, 1, 0])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1
   

def test_inference():
    """
   Test that inference returns predictions as a numpy array 
   with the same number of rows as the input data.
    """
    X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y_train = np.array([0, 1, 0, 1])
    model = train_model(X_train, y_train)
    X_test = np.array([[1, 2], [3, 4]])
    preds = inference(model, X_test)
    assert len(preds) == len(X_test)
