# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 11:52:58 2024

@author: pakornl
"""

from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd
from sklearn.feature_selection import SelectFromModel 

dataset = pd.read_csv("C:\\Users\\gluee\\Documents\\ปี4เทอม2\\Cs377\\testpython\\Diabetesv3.csv")
X = dataset.drop(columns=['Outcome'])
y = dataset['Outcome']

print(X.shape)

clf = ExtraTreesClassifier(n_estimators=200)
clf.fit(X, y)
print(clf.feature_importances_)

model = SelectFromModel(clf, prefit=True)
X_new = model.transform(X)
print(X_new.shape)

selected_features = dataset.columns[model.get_support(indices=True)]
print("Selected features:", selected_features)

