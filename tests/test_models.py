import faulthandler
import logging
import os

import pandas as pd

from futureframe.data.features import extract_target_variable, get_num_classes
from futureframe.models.cm2 import CM2Classifier

logging.basicConfig(level=logging.INFO)

faulthandler.enable()


def test_cm2():
    df = pd.read_csv(os.path.join("tests", "data", "car.csv"))
    X, y = extract_target_variable(df)
    num_class = get_num_classes(y)
    clf = CM2Classifier(num_class=num_class)
    y_pred = clf(X)
    assert y_pred.shape[0] == X.shape[0]
    assert y_pred.shape[1] == num_class
