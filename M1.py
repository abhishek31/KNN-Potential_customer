import itertools
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn import preprocessing 
from matplotlib.ticker import NullFormatter
%matplotlib inline
# we have a datasets that contains the customer information by service usage pattern 
# we need to build a model that will be used to predict class of a new or unknown case.
# The target field is cuscat that has four possible values i.e. 1-basic service, 2-E service, 3-plus service, 4-total service
# we will use k-nearest neighbour to build the model.

!wget -O teleCust1000t.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/teleCust1000t.csv

df = pd.read_csv('teleCust1000t.csv')
#df.columns
x= df[['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed',
       'employ', 'retire', 'gender', 'reside']].values
y= df['custcat'].values
# Normalizing data -it give data zero mean and unit variance
X= preprocessing.StandardScaler().fit(x).transform(x.astype(float))
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=4)
# Classification (KNN)
from sklearn.neighbors import KNeighborsClassifier 
k=4
neigh= KNeighborsClassifier(n_neighbors = k).fit(X_train, y_train)
# Predicting the value

yhat = neigh.predict(X_test)
from sklearn import metrics 
print(" Train set accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print(" Test set accuracy:  ", metrics.accuracy_score(y_test, neigh.predict(X_test)))
print('the training set', X_train.shape, y_train.shape)
print('the test set', X_test.shape, y_test.shape)

# FInding the best value of K
ks=100
mx = np.zeros((ks-1))
ConfustionMx =[];
for n in range(1,ks):
    #train model and predict 
    neigh =KNeighborsClassifier(n_neighbors = n).fit(X_train, y_train)
    yhat=neigh.predict(X_test)
    mx[n-1] = metrics.accuracy_score(y_test, yhat)
mx

print('The best accuracy was', mx.max()*100,'%', 'at k =', mx.argmax()+1)

plt.plot(range(1,ks),mx,'g')
plt.xlabel('value of k')
plt.ylabel('Accuracy')
plt.show()