import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd

players = pd.read_csv("allsituations.csv")
fouls["position"].replace({"C" : 0, "R" : 1, "L" : 2, "D" : 3}, inplace=True)

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import svm

X_train, X_test, y_train, y_test = train_test_split(
    fouls[["exit_velocity", "predicted_zone", "camera_zone", "used_zone"]], fouls.type_of_hit, test_size=0.2, random_state=109)  # 80% training and 20% test
