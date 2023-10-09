import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder


def load_data(path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Args:
        path: The path to the CSV file.

    Returns:
        A Pandas DataFrame containing the data from the CSV file.
    """
    # Read the data from the CSV file.
    data = pd.read_csv(path)

    return data


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the data by removing spaces in column names,
    converting invalid dtypes in str dtypes to np.nan and
    removing them, and removing values '?' in data.

    Args:
        df: The Pandas DataFrame to be cleaned.

    Returns:
        The cleaned Pandas DataFrame.
    """
    # Remove space in column name
    print(f"Data contains {df.shape[0]} rows")
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace("-", "_")
    object_cols = df.select_dtypes('object').columns
    # Update invalid dtype in str dtypes to np.nan and remove it
    df.loc[:, object_cols] = df[object_cols].applymap(
        lambda x: x.strip() if isinstance(x, str) else np.nan
    )
    df = df.dropna(subset=object_cols, how='any').reset_index(drop=True)
    # Remove value '?' in data
    mask = df.applymap(
        lambda x: x == "?" if isinstance(x, str) else False
    ).any(axis=1)
    df = df[~mask].reset_index(drop=True)
    print(f"Data after cleaned contains {df.shape[0]} rows")
    return df


def process_data(
    X, categorical_features=[], label=None, training=True, encoder=None, lb=None
):
    """
    Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Note: depending on the type of model used, you may want to add in functionality that
    scales the continuous data.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the encoder passed
        in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the binarizer
        passed in.
    """

    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1)

    if training is True:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()
    else:
        X_categorical = encoder.transform(X_categorical)
        try:
            y = lb.transform(y.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            pass

    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X, y, encoder, lb
