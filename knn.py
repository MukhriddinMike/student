import sklearn
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model


# well open data
data = pd.read_csv('car.data')
print(data.head())

#now we are converting our string data to numeric

le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data['buying']))
