from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from collections import OrderedDict
import numpy as np
import pandas as pd
import matplotlib.pyplot as mp
import random

# Imported the Heart.csv data file
filename = 'Heart.csv'
data = pd.read_csv(filename, header = 0)

# Dropping missing values
data = data.dropna()


# Data pre-processing
ChestPain = {'asymptomatic' : 0, 'nonanginal' : 1, 'typical' : 2, 'nontypical' : 3}
Thal = {'fixed' : 0, 'normal' : 1, 'reversable' : 2}
AHD = {'Yes' : 0, 'No' : 1}

data.ChestPain = [ChestPain[item] for item in data.ChestPain]
data.Thal = [Thal[item] for item in data.Thal]
data.AHD = [AHD[item] for item in data.AHD]

# Splitting the dataset into Training and Testing data
train_data, test_data = train_test_split(data, test_size = 0.3)
X = train_data[['Age', 'Sex', 'ChestPain', 'RestBP', 'Chol', 'Fbs', 'RestECG', 'MaxHR', 'ExAng', 'Oldpeak', 'Slope', 'Ca', 'Thal']]
Y = train_data[['AHD']]
Y = np.ravel(Y)
X1 = test_data[['Age', 'Sex', 'ChestPain', 'RestBP', 'Chol', 'Fbs', 'RestECG', 'MaxHR', 'ExAng', 'Oldpeak', 'Slope', 'Ca', 'Thal']]
Y1 = test_data[['AHD']]
sup = svm.LinearSVC()
sup.fit(X,Y)
print(sup.score(X,Y))
print(sup.score(X1,Y1))


# Applying Linear Support Vector Machine Classifier and plotting accuracy of the algorithm
c = [1e-2, 1, 1e2]
train_plotting = []
test_plotting = []
for i in c:
    supp = svm.LinearSVC(C = i)
    supp.fit(X,Y)
    train_accuracy = supp.score(X,Y) 
    test_accuracy = supp.score(X1,Y1) # Accuracy of Linear SVM
    train_plotting.append((i, train_accuracy))
    test_plotting.append((i, test_accuracy))
x_train,y_train = zip(*train_plotting)
x_test, y_test = zip(*test_plotting)
mp.plot(x_train,y_train, label = 'Training set')
mp.plot(x_test, y_test, label = 'Test set')
mp.xlabel("C")
mp.ylabel("Accuracy")
mp.legend()
mp.show()


# Applying SVM with polynomial kernel and plotting the accuracy
suppoly = svm.SVC(kernel = 'poly', degree = 4)
suppoly.fit(X,Y)
print(suppoly.score(X1,Y1)) 


#Applying SVM with RBF kernel and plotting the accuracy
suprad = svm.SVC(kernel = 'rbf')
suprad.fit(X,Y)
print(suprad.score(X1,Y1))