# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 21:54:47 2024

@author: gluee
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

url = "C:\\Users\\gluee\\Documents\\ปี4เทอม2\\Cs377\\testpython\\Diabetesv3.csv"
# Assign column names to the dataset
names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
# Read dataset to pandas dataframe, skip first row which contains column names
dataset = pd.read_csv(url, skiprows=1, names=names)

# Keep only 'age' and 'Glucose' columns
X = dataset[['Age', 'Glucose']].values
y = dataset['Outcome'].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

###Normalize 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

from sklearn.model_selection import cross_val_score

# Perform 20-fold cross validation
scores = cross_val_score(classifier, X, y, cv=10)

# Print the mean and standard deviation of the cross validation scores
print("Mean accuracy:", scores.mean())
print("Standard deviation of accuracy:", scores.std())

error = []

# Calculating error for K values between 1 and 40
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))
    
plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()

train_score = classifier.score(X_train, y_train)
test_score = classifier.score(X_test, y_test)

print("Train score:", train_score)
print("Test score:", test_score)
