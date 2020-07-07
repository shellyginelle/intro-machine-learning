#------------------------------------
# Imports
#------------------------------------
import sys
print("Python version:", sys.version)
import pandas as pd
print("pandas version:", pd.__version__)
import matplotlib
print("matplotlib version:", matplotlib.__version__)
import numpy as np
print("numpy version:", np.__version__)
import scipy as sp
print("scipy version:", sp.__version__)
import IPython
print("IPython version:", IPython.__version__)
import sklearn
print("sklearn version:", sklearn.__version__)
import mglearn

#------------------------------------
# Get Data
#------------------------------------
from sklearn.datasets import load_iris
iris_dataset = load_iris()
print("Keys of iris_dataset:\n", iris_dataset.keys())
print(iris_dataset['DESCR'][:193] + "\n...")

#------------------------------------
# Data & Target Descriptions
#------------------------------------
print("Target names:", iris_dataset['target_names'])
print("Feature names:", iris_dataset['feature_names'])
print("Type of data:", type(iris_dataset['data']))
print("Shape of data:", iris_dataset['data'].shape)
print("First five rows of data:\n", iris_dataset['data'][:5])

print("Type of data:", type(iris_dataset['target']))
print("Shape of target:", iris_dataset['target'].shape)
print("Target:\n", iris_dataset['target'])

#------------------------------------
# Train Test Split
#------------------------------------
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0
)
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

#------------------------------------
# Data Review
#------------------------------------
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
pd.plotting.scatter_matrix(
    iris_dataframe, c=y_train, figsize=(15, 15),
    marker='o', hist_kwds={'bins': 20}, s=60,
    alpha=.8, cmap=mglearn.cm3)

#------------------------------------
# Creating the Model
#------------------------------------
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, y_train)

KNeighborsClassifier(
    algorithm='auto', leaf_size=30, metric='minkowski',
    metric_params=None, n_jobs=None, n_neighbors=1, p=2,
    weights='uniform'
)

#------------------------------------
# Making Predictions
#------------------------------------
X_new = np.array([[5, 2.9, 1, 0.2]])
print('X_new.shape:', X_new.shape)

prediction = knn.predict(X_new)
print('Prediction:', prediction)
print('Predicted target name:',
    iris_dataset['target_names'][prediction])

#------------------------------------
# Evaluation
#------------------------------------
y_pred = knn.predict(X_test)
print('Test set predictions:\n', y_pred)
print('Test set score: {:.2f}'.format(np.mean(y_pred == y_test)))
print('Test set score: {:.2f}'.format(knn.score(X_test, y_test)))