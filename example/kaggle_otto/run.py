import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


# Using preprossess code from kaggle_otto_nn.py directly

def load_data(path, train=True):
    df = pd.read_csv(path)
    X = df.values.copy()
    if train:
        np.random.shuffle(X) # https://youtu.be/uyUXoap67N8
        X, labels = X[:, 1:-1].astype(np.float32), X[:, -1]
        return X, labels
    else:
        X, ids = X[:, 1:].astype(np.float32), X[:, 0].astype(str)
        return X, ids

def preprocess_data(X, scaler=None):
    if not scaler:
        scaler = StandardScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler

def preprocess_labels(labels, encoder=None, categorical=True):
    if not encoder:
        encoder = LabelEncoder()
        encoder.fit(labels)
    y = encoder.transform(labels).astype(np.int32)
    ymax = max(y)
    if categorical:
        y_label = np.zeros((y.shape[0], ymax + 1), dtype=np.float32)
        for i in xrange(y.shape[0]):
            y_label[i][y[i]] = 1.0
    return y_label, encoder, y

X, labels = load_data('train.csv', train=True)
X, scaler = preprocess_data(X)
y, encoder, yy = preprocess_labels(labels)

data = np.hstack((yy.reshape(X.shape[0], 1), X))
np.savetxt("processed.csv", data, delimiter=",", fmt="%.5f")

import os
os.system("sed -n '1, 50000p' processed.csv > tr.csv")
os.system("sed -n '50000, 70000p' processed.csv > va.csv")

