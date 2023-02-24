import os
import pandas as pd
import rampwf as rw
from sklearn.metrics import f1_score
from sklearn.model_selection import ShuffleSplit

_features_name = ["designation", "description", "productid", "imageid"]

_target_column_name = "prdtypecode"

_prediction_label_names = [
    1,
    4,
    5,
    6,
    13,
    114,
    116,
    118,
    128,
    132,
    156,
    192,
    194,
    206,
    222,
    228,
    1281,
    1301,
    1302,
    2403,
    2462,
    2522,
    2582,
    2583,
    2585,
    2705,
    2905,
]

problem_title = "Rakuten France Multimodal Product Data Classification"

Predictions = rw.prediction_types.make_multiclass(label_names=_prediction_label_names)

workflow = rw.workflows.Classifier()

score_types = [
    rw.score_types.f1_above(name="F1_score"),
    rw.score_types.Accuracy(name="acc"),
]


def get_cv(X, y):
    cv = ShuffleSplit(n_splits=8, test_size=0.2, random_state=50)
    return cv.split(X, y)


def _read_data(path, f_name):
    data_x = pd.read_csv(os.path.join(path, "data", f_name[0]))
    X_df = data_x[_features_name]
    data_y = pd.read_csv(os.path.join(path, "data", f_name[1]))
    y_array = data_y[_target_column_name]
    return X_df, y_array


def get_train_data():
    path = "."
    f_name = ["X_train", "y_train"]
    return _read_data(path, f_name)


def get_test_data():
    path = "."
    f_name = ["X_test", "y_test"]
    return _read_data(path, f_name)
