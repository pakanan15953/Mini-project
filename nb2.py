# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 21:55:35 2024

@author: gluee
"""

import pandas as pd
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler

# Load dataset
dataset = pd.read_csv("C:\\Users\\gluee\\Documents\\ปี4เทอม2\\Cs377\\testpython\\Diabetesv3.csv")

# Replace 0 with 1 in all columns except "Outcome" and "Pregnancies"
cols_to_replace = dataset.columns.drop(["Outcome", "Pregnancies"])
dataset[cols_to_replace] = dataset[cols_to_replace].replace(0, 1)

# Keep only 'age' and 'Glucose' columns
X = dataset[['Age', 'Glucose']].values
y = dataset["Outcome"].values

# Normalize data
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train and predict using BernoulliNB
modelBNB = BernoulliNB()
modelBNB.fit(X_train, y_train)
y_pred_BNB = modelBNB.predict(X_test)

# Train and predict using GaussianNB
modelGNB = GaussianNB()
modelGNB.fit(X_train, y_train)
y_pred_GNB = modelGNB.predict(X_test)

# Train and predict using MultinomialNB
modelMNB = MultinomialNB()
modelMNB.fit(X_train, y_train)
y_pred_MNB = modelMNB.predict(X_test)

# Confusion Matrix and Performance Metrics for BernoulliNB
print("\nConfusion Matrix for BernoulliNB:")
print(confusion_matrix(y_test, y_pred_BNB))
print("\nClassification Report for BernoulliNB:")
print(classification_report(y_test, y_pred_BNB, zero_division=0))

# Confusion Matrix and Performance Metrics for GaussianNB
print("\nConfusion Matrix for GaussianNB:")
print(confusion_matrix(y_test, y_pred_GNB))
print("\nClassification Report for GaussianNB:")
print(classification_report(y_test, y_pred_GNB, zero_division=0))

# Confusion Matrix and Performance Metrics for MultinomialNB
print("\nConfusion Matrix for MultinomialNB:")
print(confusion_matrix(y_test, y_pred_MNB))
print("\nClassification Report for MultinomialNB:")
print(classification_report(y_test, y_pred_MNB, zero_division=0))

# Train and test scores for all models
train_score_BNB = modelBNB.score(X_train, y_train)
test_score_BNB = modelBNB.score(X_test, y_test)

train_score_GNB = modelGNB.score(X_train, y_train)
test_score_GNB = modelGNB.score(X_test, y_test)

train_score_MNB = modelMNB.score(X_train, y_train)
test_score_MNB = modelMNB.score(X_test, y_test)

# Print scores for all models
print("BernoulliNB - Train score:", '{:.2f}'.format(train_score_BNB * 100))
print("BernoulliNB - Test score:", '{:.2f}'.format(test_score_BNB * 100))

print("GaussianNB - Train score:", '{:.2f}'.format(train_score_GNB * 100))
print("GaussianNB - Test score:", '{:.2f}'.format(test_score_GNB * 100))

print("MultinomialNB - Train score:", '{:.2f}'.format(train_score_MNB * 100))
print("MultinomialNB - Test score:", '{:.2f}'.format(test_score_MNB * 100))
