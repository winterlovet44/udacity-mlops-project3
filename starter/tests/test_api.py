"""
This module use to test FastAPI Backend

The test cases:
   - a GET method
   - 2 POST methods
"""

import os
import sys
from pathlib import Path
import logging

abs_path = Path(os.path.abspath(__file__))
# src_dir = os.path.join(abs_path.parent.parent, 'starter')
sys.path.append(str(abs_path.parent.parent))
print(sys.path)

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_get_method():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome!!!"}

def test_post_method():
    """Test the predict endpoint with label <=50K."""
    sample_data = {
        'age': 54,
        'workclass': 'Private',
        'fnlgt': 302146,
        'education': 'HS-grad',
        'education_num': 9,
        'marital_status': 'Separated',
        'occupation': 'Other-service',
        'relationship': 'Unmarried',
        'race': 'Black',
        'sex': 'Female',
        'capital_gain': 0,
        'capital_loss': 0,
        'hours_per_week': 20,
        'native_country': 'United-States'
    }
    response = client.post("/predict", json=sample_data)
    assert response.status_code == 200
    assert response.json() == {"prediction": "<=50K"}

def test_post_method2():
    """Test the predict endpoint with label >50K."""
    sample_data = {
        'age': 37,
        'workclass': 'Private',
        'fnlgt': 635913,
        'education': 'Bachelors',
        'education_num': 13,
        'marital_status': 'Never-married',
        'occupation': 'Exec-managerial',
        'relationship': 'Not-in-family',
        'race': 'Black',
        'sex': 'Male',
        'capital_gain': 0,
        'capital_loss': 0,
        'hours_per_week': 60,
        'native_country': 'United-States'
    }
    response = client.post("/predict", json=sample_data)
    assert response.status_code == 200
    assert response.json() == {"prediction": ">50K"}
