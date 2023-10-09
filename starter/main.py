# Put the code for your API here.
import os
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd

from starter.ml.data import process_data
from starter.ml.model import inference, load_model
from cfg import param, cat_features

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

# Load the data and model when the module is imported, not each time the app runs.
model = load_model(path=param['MODEL_PATH'])
lb = load_model(path=param['LABEL_PATH'])
encoder = load_model(path=param['ENCODER_PATH'])

# Initialize backend
app = FastAPI()


class DataInput(BaseModel):
    workclass: str = Field(default="State-gov")
    education: str = Field(default="HS-grad")
    marital_status: str = Field(default="Married-civ-spouse")
    occupation: str = Field(default="Prof-specialty")
    relationship: str = Field(default="Husband")
    race: str = Field(default="White")
    sex: str = Field(default="Male")
    native_country: str = Field(default="United-States")
    age: int = Field(default=25)
    fnlgt: int = Field(default=338409)
    education_num: int = Field(default=13)
    capital_gain: int = Field(default=0)
    capital_loss: int = Field(default=0)
    hours_per_week: int = Field(default=40)


@app.get("/")
async def root():
    return {
        "message": "Welcome!!!"
    }


@app.post("/predict")
async def predict(data: DataInput):
    try:
        dat = {
            "age": [data.age],
            "workclass": [data.workclass],
            "fnlgt": [data.fnlgt],
            "education": [data.education],
            "education_num": [data.education_num],
            "marital_status": [data.marital_status],
            "occupation": [data.occupation],
            "relationship": [data.relationship],
            "race": [data.race],
            "sex": [data.sex],
            "capital_gain": [data.capital_gain],
            "capital_loss": [data.capital_loss],
            "hours_per_week": [data.hours_per_week],
            "native_country": [data.native_country],
        }

        df = pd.DataFrame(dat)

        # Ensure categorical fields are treated as strings in DataFrame
        df[cat_features] = df[cat_features].astype(str)

        # Use the correct order of categorical fields for the LabelBinarizer
        X, _, _, _ = process_data(
            df, categorical_features=cat_features, encoder=encoder, lb=lb, training=False)

        prediction = inference(model, X)

        # Inverse transform the prediction using the LabelBinarizer
        pred = lb.inverse_transform(prediction)

        return {
            "prediction": pred[0]
        }

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    is_on_heroku = 'DYNO' in os.environ

    import uvicorn

    config = uvicorn.Config(
        "main:app",
        host="0.0.0.0" if is_on_heroku else "127.0.0.1",
        reload=not is_on_heroku,
        port=int(os.environ.get('PORT', 8000)),
        log_level="info"
    )
    server = uvicorn.Server(config)
    server.run()
