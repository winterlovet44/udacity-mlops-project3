# Script to train machine learning model.
import os

from sklearn.model_selection import train_test_split
# Add the necessary imports for the starter code.
from starter.ml.model import train_model, compute_model_metrics, save_model
from starter.ml.data import load_data, clean_data, process_data
from cfg import param, cat_features


# Add code to load in the data.
data = load_data(path=param["DATA_PATH"])
data = clean_data(df=data)

# Save cleaned data
data.to_csv(param["CLEANED_DATA_PATH"], index=False)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

# cat_features = [
#     "workclass",
#     "education",
#     "marital-status",
#     "occupation",
#     "relationship",
#     "race",
#     "sex",
#     "native-country",
# ]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Save encoder and labeler for future usage
save_model(encoder, param["ENCODER_PATH"])
save_model(lb, param["LABEL_PATH"])

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    X=test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb
)
# Train and save a model.
print("Start to train ML model...")
model = train_model(X_train, y_train)
print("Predict label of test data...")
y_pred = model.predict(X_test)
print("Evaluate the model...")
prec, recall, fbeta = compute_model_metrics(y_test, y_pred)
print(f"Precision: {prec:.2f}, Recall: {recall:.2f}, Fbeta: {fbeta:.2f}")
print(f"Save the model to {param['MODEL_PATH']}")
save_model(model, param["MODEL_PATH"])
