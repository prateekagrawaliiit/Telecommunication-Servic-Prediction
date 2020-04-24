# DEPENDENCIES

import itertools
import numpy as np 
import pandas as pd 
import pylab as py 
import matplotlib.pyplot as plt 
from matplotlib.ticker import NullFormatter
import matplotlib.ticker as ticker
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
"%matplotlib inline"


# IMPORTING DATASET
df = pd.read_csv("teleCust1000t.csv")
# print(df.head())  

#VISUALIZING DATA
# print(df['custcat'].value_counts())

# df.hist(column='income', bins=50)
# plt.show()

# print(df.columns)

X = df[['region','tenure','age','marital','address','income','ed','employ','retire','gender','reside']].values
y=df['custcat'].values

#DATA NORMALIZATION (Usually a better practice for KNN also gives us 0 mean and unit variance. Standard Normal Form)
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
# print(X)
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
""" print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape) """   

k =38
neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train,y_train)

y_pred= neigh.predict(X_test)
# print(y_pred) 

print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, y_pred))


#HOW TO FIND OUT THE BEST K and BEST ACCURACE : 
""" 

Ks = 50
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
for n in range(1,Ks):
    
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

# print(mean_acc)

plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbours (K)')
plt.tight_layout()
plt.show()

print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1)  """