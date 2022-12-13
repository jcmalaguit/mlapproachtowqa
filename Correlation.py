from pandas import read_csv
from pandas import set_option
import seaborn as sns

set_option('display.max_columns', None)
set_option('display.max_rows', None)

laguna = 'ForCorrelation.csv'
names = ['pH', 'ammonia', 'nitrate', 'BOD', 'DO', 'fecal coliform', 'class']
dataset = read_csv(laguna, names=names)

# class distribution
print(dataset.groupby('class').size())

corr = dataset.corr()
print(corr)

sns.heatmap(corr, xticklabels=corr.columns,
        yticklabels=corr.columns, annot=True)
