import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression

#loading the breast cancer data set from scikit-learn into a variable
cancer_data = load_breast_cancer()

#printing the various aspects of the breast cancer data set
print(cancer_data.keys())
print(cancer_data['DESCR'])
cancer_data['data'].shape #records the total number of data points in the data set
cancer_data['feature_names']

#creating a DataFrame using pandas to store the breast cancer data set
df = pd.DataFrame(cancer_data['data'], columns = cancer_data['feature_names'])
print(df.head)
cancer_data['target']
cancer_data['target'].shape
cancer_data['target_names']
df['target'] = cancer_data['target']
print(df.head())

#creating the feature matrix for the logistic regression model using NumPy
x = df[cancer_data.feature_names].values

#creating the target array for the logistic regression model using NumPy
y = df['target'].values

#instantiating the logistic regression model object and performing the logistic regression analysis
model = LogisticRegression(solver = 'liblinear') #the solver is specified since the default solver produces a Convergence Warning which can be solved either by increasing iterations or changing solvers
model.fit(x, y)

#using the model to predict the first data point in the set
print(model.predict([x[0]]))

#calculating the model's accuracy percentage
print(model.score(x, y))