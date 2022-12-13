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


laguna = 'Data.csv'
names = ['pH', 'ammonia', 'nitrate', 'BOD', 'DO', 'fecal coliform', 'class']
dataset = read_csv(laguna, names=names)
print(dataset.shape)

#descriptions
print(dataset.describe())

# class distribution
print(dataset.groupby('class').size())

# split dataset
array = dataset.values
X = array[:,0:6]
y = array[:,6]
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.30, random_state=1, shuffle=True)

# print('Training Features Shape:', X_train.shape)
# print('Training Labels Shape:', Y_train.shape)
# print('Testing Features Shape:', X_validation.shape)
# print('Testing Labels Shape:', Y_validation.shape)

# ML implementation
models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('NB', GaussianNB()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('SVM', SVC(gamma='auto')))

# k-fold cross-validation
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10)
    cv_results1 = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    cv_results2 = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='precision_macro')
    cv_results3 = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='recall_macro')
    cv_results4 = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='f1_macro')
    names.append(name)
    print('%s: %f %f %f %f ' % (name, cv_results1.mean(), cv_results2.mean(), cv_results3.mean(), cv_results4.mean()))

# testing
for name, model in models:
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    names.append(name)
    print(name)
    print(accuracy_score(Y_test, predictions))
    print(confusion_matrix(Y_test, predictions))
    print(classification_report(Y_test, predictions, zero_division=0))
