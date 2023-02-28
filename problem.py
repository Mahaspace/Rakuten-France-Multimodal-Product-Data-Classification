import os
import pandas as pd
import rampwf as rw
from sklearn.metrics import f1_score
from sklearn.model_selection import ShuffleSplit

_features_name = ["designation", "description", "productid", "imageid"]

_target_column_name = "prdtypecode"

_prediction_label_names = [
    1320,
    2582,
    2583,
    1160,
    1560,
    2585,
    1920,
    2280,
    2705,
    1300,
    2060,
    2403,
    2522,
    40,
    1302,
    1280,
    1140,
    50,
    2462,
    2220,
    1180,
    1301,
    2905,
    1940,
    1281,
    60,
    10,
]


problem_title = "Rakuten France Multimodal Product Data Classification"

Predictions = rw.prediction_types.make_multiclass(label_names=_prediction_label_names)

workflow = rw.workflows.Classifier()

score_types = [
    rw.score_types.F1Above(),
    rw.score_types.Accuracy(name="acc"),
]


def get_cv(X, y):
    cv = ShuffleSplit(n_splits=3, test_size=0.2, random_state=42)
    return cv.split(X, y)


def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, "data", "public", f_name))
    X_df = data[_features_name]
    y_array = data[_target_column_name]
    return X_df.to_numpy(), y_array


def get_train_data(path="."):
    path = "."
    f_name = "train.csv"
    return _read_data(path, f_name)


def get_test_data(path="."):
    path = "."
    f_name = "test.csv"
    return _read_data(path, f_name)
