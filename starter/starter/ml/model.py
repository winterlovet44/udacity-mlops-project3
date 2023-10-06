import os
import joblib

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score
from starter.ml.data import process_data


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train) -> RandomForestClassifier:
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model: RandomForestClassifier, X: np.array) -> int:
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : Sklearn RandomForestClassifier model
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : int
        Predictions from the model.
    """
    return model.predict(X)


def save_model(model: RandomForestClassifier, filename: str, model_dir: str='model'):
    """
    Save a Scikit-learn model to a file using joblib.

    Parameters
    ----------
    model : Scikit-learn model
        The Scikit-learn model to be saved.
    filename : str
        The name of the file to save the model to.
    model_dir: str, default: model
        The folder contains all model file

    Returns
    -------
    None.

    """
    path = os.path.join(model_dir, filename)
    joblib.dump(model, path)

def load_model(path: str) -> RandomForestClassifier:
    """
    Load a Scikit-learn model from a file using joblib.

    Args:
        path: str
        The name of the file that contains the model.

    Returns:
        The Scikit-learn model that was loaded from the file.
    """

    return joblib.load(path)

def slices_performance(df, categorical_features, model, encoder, lb, label='salary'):
    """
    Perform evaluation for each category in categorycal features.

    Parameters
    df: pd.DataFrame
        Dataset for training
    categorical_features: list
        List categorical feature
    model: RandomForestClassifier
        Trained model
    encoder: OneHotEncoder
        One hot encoder model for encode cateogorical feature
    lb: LabelBinarizer
        Label binarry encoder
    label: str
        Column name of label
    
    Returns
    ---------
    Dictionary contains performance of each category
    """
    result = {}
    for feature in categorical_features:
        for val in df[feature].unique():
            mask = df[df[feature] == val]
            X_test, y_test, encoder, lb = process_data(
                mask,
                categorical_features=categorical_features,
                label=label,
                training=False,
                encoder=encoder,
                lb=lb
            )
            preds = model.predict(X_test)
            prec, recall, fbeta = compute_model_metrics(y_test, preds)
            result[f"{feature}_{val}"] = f"Precision: {prec:.1f}, Recall: {recall:.1f}, fbeta: {fbeta:.1f}"
    return result
