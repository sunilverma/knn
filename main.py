import pandas as pd
# This library provides data structures and data analysis tools, which make working with structured data (like data
# in SQL tables or Excel spreadsheets) easy in Python.
import numpy as np
# This is a library for the Python programming language, adding support for large, multidimensional arrays and
# matrices, along with a large collection of high-level mathematical functions to operate on these arrays.
import matplotlib.pyplot as plt
# This is a plotting library for creating static, animated, and interactive visualizations in Python. pyplot is a
# collection of functions that make matplotlib work like MATLAB.
import seaborn as sns
# This is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing
# attractive and informative statistical graphics.

from sklearn.model_selection import train_test_split
# Importing necessary libraries and modules from sklearn for model building and evaluation, and explainerdashboard
# for model interpretation.

from sklearn.neighbors import KNeighborsClassifier
# Loading a CSV file named “wisc_bc_data.csv” into a pandas DataFrame.

from explainerdashboard import ClassifierExplainer, ExplainerDashboard

df = pd.read_csv("wisc_bc_data.csv")
# Dropping the ‘id’ column from the DataFrame.

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Mapping the ‘M’ (malignant) and ‘B’ (benign) values in the ‘diagnosis’ column to 1 and 0, respectively.

df.drop(['id'], axis=1, inplace=True)
# Creating a pairplot of ‘radius_mean’, ‘texture_mean’, and ‘perimeter_mean’ variables, colored by the ‘diagnosis’
# variable.

ctypes = {'M': 1, 'B': 0}

df['diagnosis'] = df['diagnosis'].map(ctypes)

X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Create explainer instance
explainer = ClassifierExplainer(knn, X_test, y_test)

# Create the dashboard
db = ExplainerDashboard(explainer)

# Run the dashboard
db.run(port=8050)
