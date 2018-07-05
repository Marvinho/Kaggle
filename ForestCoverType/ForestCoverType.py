# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 11:58:42 2018

@author: MRVN
"""
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

traindata = pd.read_csv("./input/train.csv")
testdata = pd.read_csv("./input/test.csv")



X = traindata.drop(["Id", "Cover_Type"], axis = 1)
y = traindata.Cover_Type
testX = testdata.drop(["Id"], axis = 1)

trainX, valX, trainY, valY = train_test_split(X, y, random_state = 0)

"""
model = RandomForestClassifier(n_estimators=100, 
                               criterion="gini", 
                               max_depth=None, 
                               min_samples_split=2, 
                               min_samples_leaf=1, 
                               min_weight_fraction_leaf=0.0,
                               max_features="auto", 
                               max_leaf_nodes=None, 
                               min_impurity_decrease=0.0, 
                               min_impurity_split=None, 
                               bootstrap=True, 
                               oob_score=False, 
                               n_jobs=1, 
                               random_state=None, 
                               verbose=0, 
                               warm_start=False, 
                               class_weight=None)

"""
"""
model = BaggingClassifier(KNeighborsClassifier(), 
                                  max_samples=0.5, 
                                  max_features=0.5)
"""

model = XGBClassifier(learning_rate = 0.2, 
                      n_estimators = 500, 
                      gamma = 0, 
                      max_depth = 10, 
                      subsample = 0.8
                      )

"""
model = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=True)

model = LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='crammer_singer', penalty='l2', random_state=None, tol=0.0001,
     verbose=0)
"""
"""
model = KNeighborsClassifier()
"""
"""
model = LogisticRegression(penalty="l2", 
                           dual=False, 
                           tol=0.0001, 
                           C=1.0, 
                           fit_intercept=True, 
                           intercept_scaling=1, 
                           class_weight=None, 
                           random_state=None, 
                           solver="liblinear", 
                           max_iter=100, 
                           multi_class="ovr", 
                           verbose=0, 
                           warm_start=False, 
                           n_jobs=-1)
"""
model.fit(X, y)

valPreds = model.predict(valX)


print("Making predictions for the following:")
print(valY[0:20])
print("The predictions are")
print(valPreds[0:20])

scores = cross_val_score(model, X, y, scoring = "accuracy", cv = 3)

print(scores)
"""
testPreds = model.predict(testX)
submission = pd.DataFrame({'Id': testdata.Id, 'Cover_Type': testPreds})
# you could use any filename. We choose submission here
submission.to_csv('submission.csv',columns=["Cover_Type","Id"], index=False)
"""
