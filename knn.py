import sklearn
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model
import pickle

# well open data
data = pd.read_csv('car.data')
print(data.head())

#now we are converting our string data to numeric

le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data['buying']))
maint = le.fit_transform(list(data['maint']))
cls = le.fit_transform(list(data['class']))
safety = le.fit_transform(list(data['safety']))
door =le.fit_transform(list(data['door']))
persons = le.fit_transform(list(data['persons']))
lug_boot = le.fit_transform(list(data['lug_boot']))


# Now we gonna do label and feature

X = list(zip(buying, maint, door, persons, lug_boot, safety)) # feature
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y, test_size=0.1)

model =KNeighborsClassifier(n_neighbors=9)
# n_neighbors is key word and finds n number of neighbors

model.fit(x_train, y_train)

acc = model.score(x_test, y_test)



best = 0
for _ in range(1000):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y, test_size=0.1)


    model =KNeighborsClassifier(n_neighbors=9)
    # n_neighbors is key word and finds n number of neighbors

    model.fit(x_train, y_train)

    acc = model.score(x_test, y_test)

    #print(acc)

    if acc > best:
        print(acc)

#print(acc)

predicted = model.predict(x_test)
names = ['unacc', 'acc', 'good', ' vgood']

for x in range(len(predicted)):
    print("Predict :  ",names[predicted[x]], "Data: ", x_test[x], "Actual: ", names[y_test[x]])
   # n =model.kneighbors([x_test[x]], 9 , True)
   #  print('N: ', n)
