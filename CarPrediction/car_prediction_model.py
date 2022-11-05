#Car Prediction 

#Importing the libraries
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt

#Importing Dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

#Fitting the classifier to the Training Set
from sklearn.svm import SVC
classifier = SVC(C = 3, kernel = 'rbf', gamma = 0.8, random_state = 0)
#classifier.fit(X_train, y_train)
classifier.fit(X, y)

#Save your model
import joblib
joblib.dump(classifier, 'car_prediction_model.pkl')
print("Model dumped!")

#Load the model that you just saved
classifier = joblib.load('car_prediction_model.pkl')
