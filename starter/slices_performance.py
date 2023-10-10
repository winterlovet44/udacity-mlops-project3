# Script to train machine learning model.
import json

from sklearn.model_selection import train_test_split
# Add the necessary imports for the starter code.
from starter.ml.model import load_model, slices_performance
from starter.ml.data import load_data
from cfg import param, cat_features


# Add code to load in the data.
data = load_data(path=param["CLEANED_DATA_PATH"])

# Optional enhancement, use K-fold cross validation instead of a train-test split.
_, test = train_test_split(data, test_size=0.20)

model = load_model(path=param['MODEL_PATH'])
lb = load_model(path=param['LABEL_PATH'])
encoder = load_model(path=param['ENCODER_PATH'])

result = slices_performance(
    df=test, categorical_features=cat_features, model=model, encoder=encoder, lb=lb)

with open("slice_output.txt", "w") as f:
    json.dump(result, f, indent=4)
