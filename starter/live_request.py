"""
Test the live prediction endpoint on Render
"""
import requests
import logging

logging.basicConfig(filename='./test_render.log',
                    level=logging.INFO, format="%(asctime)-15s %(message)s")


sample = {
    'age': 27,
    'workclass': 'Private',
    'fnlgt': 201872,
    'education': 'Some-college',
    'education_num': 10,
    'marital_status': 'Married-civ-spouse',
    'occupation': 'Sales',
    'relationship': 'Husband',
    'race': 'White',
    'sex': 'Male',
    'capital_gain': 0,
    'capital_loss': 0,
    'hours_per_week': 50,
    'native_country': 'United-States',
}


app_url = "https://udacity.onrender.com/predict"

r = requests.post(app_url, json=sample)
print(r.status_code)
print(r.json())

logging.info("Testing live request")
logging.info(f"Status code: {r.status_code}")
logging.info(f"Response body: {r.json()}")
