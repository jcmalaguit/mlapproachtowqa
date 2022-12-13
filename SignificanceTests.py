from pandas import read_csv
from scipy.stats import levene
from scipy.stats import ttest_ind

default = read_csv('default.csv', header=None)
mlcorr = read_csv('ml-correlation.csv', header=None)
mlinfogain = read_csv('ml-infogain.csv', header=None)
summary = read_csv('summary.csv', header=None)

array = default.values
array2 = mlcorr.values
array3 = mlinfogain.values
array4 = summary.values

defaultF1DT = array[:,2]
print(defaultF1DT)
print(defaultF1DT.mean())

mlcorrF1DT = array2[:,2]
print(mlcorrF1DT)
print(mlcorrF1DT.mean())

mlinfogainF1DT = array3[:,2]
print(mlinfogainF1DT)
print(mlinfogainF1DT.mean())

#levene's test
levene1 = levene(defaultF1DT, mlcorrF1DT, center = 'mean')
levene2 =  levene(defaultF1DT, mlinfogainF1DT, center = 'mean')
print(levene1, levene2)

#t-test
ttest1 = ttest_ind(defaultF1DT, mlcorrF1DT, equal_var=True, alternative="greater")
ttest2 = ttest_ind(defaultF1DT, mlinfogainF1DT, equal_var=True, alternative="less")
print(ttest1, ttest2)

defaultAcc = array4[:,8]
print(defaultAcc)
print(defaultAcc.mean())

corrAcc = array4[:,24]
print(corrAcc)
print(corrAcc.mean())

infoAcc = array4[:,40]
print(infoAcc)
print(infoAcc.mean())

#levene's test
levene3 = levene(defaultAcc, corrAcc, center = 'mean')
levene4 =  levene(defaultAcc, infoAcc, center = 'mean')
print(levene3, levene4)

#t-test
ttest3 = ttest_ind(defaultAcc, corrAcc, equal_var=True, alternative="greater")
ttest4 = ttest_ind(defaultAcc, infoAcc, equal_var=True, alternative="less")
ttest5 = ttest_ind(corrAcc, infoAcc, equal_var=True, alternative="less")
print(ttest3, ttest4, ttest5)
