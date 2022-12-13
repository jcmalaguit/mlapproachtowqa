from pandas import read_csv
from sklearn.model_selection import train_test_split
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd


laguna = 'ForCorrelation.csv'
names = ['pH', 'ammonia', 'nitrate', 'BOD', 'DO', 'fecal coliform', 'class']
dataset = read_csv(laguna, names=names)
# print(dataset.shape)

#descriptions
# print(dataset.describe())

# class distribution
# print(dataset.groupby('class').size())

# split dataset
array = dataset.values
X = array[:,0:6]
y = array[:,6]
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.30, random_state=1, shuffle=True)

# print('Training Features Shape:', X_train.shape)
# print('Training Labels Shape:', Y_train.shape)
# print('Testing Features Shape:', X_test.shape)
# print('Testing Labels Shape:', Y_test.shape)
# random oversampling training data
# print(Counter(Y_train))
oversample = RandomOverSampler(sampling_strategy="minority")
X_over, Y_over = oversample.fit_resample(X_train, Y_train)
# print(Counter(Y_over))

oversample1 = RandomOverSampler(sampling_strategy="minority")
X_over1, Y_over1 = oversample.fit_resample(X_over, Y_over)
# print(Counter(Y_over1))

oversample2 = RandomOverSampler(sampling_strategy="minority")
X_over2, Y_over2 = oversample.fit_resample(X_over1, Y_over1)
# print(Counter(Y_over2))

oversample3 = RandomOverSampler(sampling_strategy="minority")
X_over3, Y_over3 = oversample.fit_resample(X_over2, Y_over2)
# print(Counter(Y_over3))

df = pd.DataFrame(X_over3,Y_over3)
print(df)

df.to_csv('ForCorrelation.csv')
