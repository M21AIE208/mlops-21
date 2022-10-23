from sklearn import datasets
from sklearn import svm
from utils import (
    preprocess_digits,
    train_dev_test_split,
    data_viz,
)
import sys
from joblib import dump, load
import os


def test_gorund_truth_labels():
    digits = datasets.load_digits()
    data_viz(digits)
    data, label = preprocess_digits(digits)
    x_train, y_train, x_dev, y_dev, x_test, y_test = train_dev_test_split(data, label, 0.8, 0.1)
    clf = load(".\models\svm_gamma=0.001_C=0.5.joblib") #loading best model
    predicted = clf.predict(x_test)
    set_1 = set(clf.classes_)
    set_2 = set(predicted)
    assert set_1==set_2  # both set of train and predicted classes must be same


def test_classifier_bias():
    digits = datasets.load_digits()
    data_viz(digits)
    data, label = preprocess_digits(digits)
    x_train, y_train, x_dev, y_dev, x_test, y_test = train_dev_test_split(data, label, 0.8, 0.1)
    clf = load(".\models\svm_gamma=0.001_C=0.5.joblib") #loading best model
    predicted = clf.predict(x_test)
    assert len(set(predicted))>1  # the classifier is not predicting only one class
