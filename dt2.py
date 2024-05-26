import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.tree import export_text

# Load the dataset
dataset = pd.read_csv("C:\\Users\\gluee\\Documents\\ปี4เทอม2\\Cs377\\testpython\\Diabetesv3.csv")

# Split the dataset into features (X) and target (y) with selected attributes
X = dataset[['Age', 'Glucose']]
y = dataset['Outcome']

# Encode the target variable if needed
lb = LabelEncoder()
y = lb.fit_transform(y)

# Decision Tree model
clf = tree.DecisionTreeClassifier(criterion="gini", max_depth=3)

# Perform 10-fold cross validation
scores = cross_val_score(clf, X, y, cv=10)

# Print the average and standard deviation of the scores
print("Average=%0.2f SD=%0.2f" % (scores.mean(), scores.std()))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# Train the model
clf.fit(X_train, y_train)

# Print the Decision Tree
tree.plot_tree(clf)
r = export_text(clf, feature_names=list(X.columns))
print(r)

# Predictions
y_pred = clf.predict(X_test)

# Evaluation
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

train_score = clf.score(X_train, y_train)
test_score = clf.score(X_test, y_test)
print("Train score: {:.2f}".format(train_score * 100))
print("Test score: {:.2f}".format(test_score * 100))