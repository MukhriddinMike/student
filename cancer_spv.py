import sklearn
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import datasets
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
cancer = datasets.load_breast_cancer()

#print("Features: ", cancer.feature_names)
#print("Labels : ", cancer.target_names)

x = cancer.data
y =cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.2)

print((x_train[:5], y_train[:5]))

#clf = svm.SVC(kernel="linear")
clf = KNeighborsClassifier(n_neighbors=11)
clf.fit(x_train, y_train)

y_pred = clf.predict((x_test))
acc = metrics.accuracy_score(y_test, y_pred)

print(acc)
