#Importing the Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing Dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values
def predict(age, salary):
        
    #Splitting the Dataset into Training Set and Test Set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    X_test[-1][0] = age
    X_test[-1][1] = salary
    
    #Feature Scailing
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    
    #Fitting the classifier to the Training Set
    from sklearn.svm import SVC
    classifier = SVC(C = 3, kernel = 'rbf', gamma = 0.8, random_state = 0)
    classifier.fit(X_train, y_train)
    
    #Predicting the Test Set result
    y_pred = classifier.predict(X_test)
    print(y_pred[-1])
    
predict(52, 90000)