"""Module contains parameter of projects."""


param = {
    "DATA_PATH": "data/census.csv",
    "CLEANED_DATA_PATH": "data/census_cleaned.csv",
    "MODEL_PATH": 'model/model_v2.pkl',
    "ENCODER_PATH": 'model/encoder.pkl',
    "LABEL_PATH": 'model/label.pkl'
}

cat_features = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]
