import numpy as np


class Classifier(object):
    def __init__(self):
        pass

    def fit(self, X_train, y):
        classes = [
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
        self.n_classes = len(classes)
        pass

    def predict_proba(self, X_train):
        
        proba = np.random.rand(len(X_train), self.n_classes)
        proba /= proba.sum(axis=1)[:, np.newaxis]

        return proba
