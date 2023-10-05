# Script to train machine learning model.
import os

from sklearn.model_selection import train_test_split
# Add the necessary imports for the starter code.
from ml.model import train_model, compute_model_metrics, save_model
from ml.data import load_data, clean_data, process_data


DATA_PATH = "../data/census.csv"
MODEL_DIR = "../model/"
MODEL_FILE_NAME = 'model_v2.pkl'
# CURRENT_TIME = datetime.now().strptime("dd-MM-yyyy")

# Add code to load in the data.
data = load_data(path=DATA_PATH)
data = clean_data(df=data)
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

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
print(f"Save the model to {os.path.join(MODEL_DIR, MODEL_FILE_NAME)}")
save_model(model, MODEL_FILE_NAME, model_dir=MODEL_DIR)
