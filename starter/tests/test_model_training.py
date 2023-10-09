"""
This module use to test model training step

The test cases:
   1. train_model
   2. Train dataset
   3. Inference result
"""

from starter.ml.model import (
    train_model,
    save_model,
    inference
)
import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path
import logging

abs_path = Path(os.path.abspath(__file__))
# src_dir = os.path.join(abs_path.parent.parent, 'starter')
sys.path.append(str(abs_path.parent.parent))
print(sys.path)


# Setup logging
logging.basicConfig(filename='tests/logging.log',
                    level=logging.INFO,
                    filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s'
                    )
# Setup data
temp_data = {
    'age': {0: 39, 1: 50, 2: 38, 3: 53, 4: 28},
    'fnlgt': {0: 77516, 1: 83311, 2: 215646, 3: 234721, 4: 338409},
    'education-num': {0: 13, 1: 13, 2: 9, 3: 7, 4: 13},
    'capital-gain': {0: 2174, 1: 0, 2: 0, 3: 0, 4: 0},
    'salary': {0: '<=50K', 1: '<=50K', 2: '<=50K', 3: '<=50K', 4: '<=50K'}
}
temp_df = pd.DataFrame(temp_data)
label = 'salary'
train_label = ['age', 'fnlgt', 'education-num', 'capital-gain']

test_sample = pd.DataFrame(
    data=[[50, 77000, 15, 2000]], columns=train_label)


def test_train_model(data=temp_df):
    try:
        _ = train_model(data[train_label], data[label])
        logging.info("Test train model succeed")
    except Exception:
        logging.error("Test train_model fail!")
        raise AssertionError("Can not train model")


def test_inference(data=temp_df):
    try:
        model = train_model(data[train_label], data[label])
        _ = inference(model, test_sample)
        logging.info("Test inference succeed!")
    except Exception as err:
        logging.error("Test inference failed.")
        raise AssertionError("Test inference function fail")


def test_save_model(data=temp_df):
    try:
        model = train_model(data[train_label], data[label])
        save_model(model, 'tests/model.pkl')
        os.remove("tests/model.pkl")
        logging.info("Test save model succeed!")
    except Exception as err:
        logging.error("Test save model failed.")
        raise AssertionError("Test save model fail")
