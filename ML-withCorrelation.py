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

laguna = 'DataCorr.csv'
names = ['pH', 'ammonia', 'BOD', 'class']
dataset = read_csv(laguna, names=names)
# print(dataset.shape)

#descriptions
# print(dataset.describe())

# class distribution
# print(dataset.groupby('class').size())

# split dataset
array = dataset.values
X = array[:,0:3]
y = array[:,3]
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

# Implementation of ML algo]rithms
models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('NB', GaussianNB()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('SVM', SVC(gamma='auto')))

# k-fold cross-validation
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10)
    cv_results1 = cross_val_score(model, X_over3, Y_over3, cv=kfold, scoring='accuracy')
    cv_results2 = cross_val_score(model, X_over3, Y_over3, cv=kfold, scoring='precision_macro')
    cv_results3 = cross_val_score(model, X_over3, Y_over3, cv=kfold, scoring='recall_macro')
    cv_results4 = cross_val_score(model, X_over3, Y_over3, cv=kfold, scoring='f1_macro')
    names.append(name)
    # print('%s: %f %f %f %f ' % (name, cv_results1.mean(), cv_results2.mean(), cv_results3.mean(), cv_results4.mean()))

# testing
for name, model in models:
    model.fit(X_over3, Y_over3)
    predictions = model.predict(X_test)
    names.append(name)
    # print(name)
    accuracy = np.round(accuracy_score(Y_test, predictions), 4)
    precision = np.round(precision_score(Y_test, predictions, average = 'macro'), 4)
    recall = np.round(recall_score(Y_test, predictions, average = 'macro'), 4)
    f1 = np.round(f1_score(Y_test, predictions, average = 'macro'), 4)
    print('%0.4f,%0.4f,%0.4f,%0.4f' %(accuracy,precision,recall,f1))
    print(accuracy_score(Y_test, predictions))
    print(confusion_matrix(Y_test, predictions))
    print(classification_report(Y_test, predictions))
