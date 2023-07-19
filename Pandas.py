import numpy as np
import pandas as pd

# 2 lists
labels = ['a', 'b', 'c']
my_list = [10, 20, 30]

# numpy array
arr = np.array([10, 20, 30])

# dictionary
d = {'a': 10, 'b': 20, 'c': 30}

# create Pandas series
# by passing my_list into pd.Series()
print(pd.Series(my_list))

# add labels to Pandas series
print(pd.Series(my_list, index=labels))

series = pd.Series(my_list, index=labels)

# we can use label or index
# they both give the same result
print(series[1])
print(series['b'])

# we can also pass dictionary into pd.Series()
print(pd.Series(d))

# pass Python built-in function to Panda
print(pd.Series([sum, print, len]))

# Pandas dataframe

# create 2 lists
rows = ['X', 'Y', 'Z']
cols = ['A', 'B', 'C', 'D', 'E']

# generate data
data = np.round(np.random.randn(3, 5), 2)

# create DataFrame
df = pd.DataFrame(data, rows, cols)
dataframe1 = df
print(df)
# another way
print(pd.DataFrame(np.round(np.random.randn(3, 5), 2),
                   ['X', 'Y', 'Z'],
                   ['A', 'B', 'C', 'D', 'E'],
                   ))

# call a Series from DataFrame
print(df['A'])
print(df['E'])

# call columns A and B
col = ['A', 'B']
print(df[col])

# call a specific element
print(df['A']['Z'])

# create a sum columns of B and C
df['B + C'] = df['B'] + df['C']
print(df)

# remove column
df.drop('B + C', axis=1, inplace=True)
print(df)
# another way to remove column
df['B + C'] = df['B'] + df['C']
df = df.drop('B + C', axis=1)
print(df)

# select a row in Pandas DataFrame
print(df.loc['X'])
# or using index number
print(df.iloc[0])

# .shape to see structure
print(df.shape)

# slicing DataFrame
print(df[['A', 'B']].loc[['X', 'Y']])

# element of DataFrame that is larger than 0.5
print(df > 0.5)

# return value where statement is True,
# and NaN where statement is False
print(df[df > 0.5])

# return boolean value in a column
print(df['B'] > 0.5)
# return value
print(df[df['B'] > 0.5])

# multiple conditions
print(df[(df['C'] < 0) & (df['A'] < 0)])

# get columns
df1 = df
print(df1.columns)
# rename columns
df1.columns = [1, 2, 3, 4, 5]
print(df1)

# create new DataFrame with missing data
df = pd.DataFrame(np.array([[1, 5, 1], [2, np.nan, 2], [np.nan, np.nan, 3]]))
df.columns = ['A', 'B', 'C']
print(df)

# remove any rows that contain NaN value
print(df.dropna())
# column
print(df.dropna(axis=1))

# replace missing values with average of the data
print(df.fillna(df.mean()))

# replace missing values with average of the column
print(df.fillna(df['A'].mean()))

# group by
df = pd.DataFrame([['Google', 'Sam', 200],
                   ['Google', 'Charlie', 120],
                   ['Salesforce', 'Ralph', 125],
                   ['Salesforce', 'Emily', 250],
                   ['Adobe', 'Rosalynn', 150],
                   ['Adobe', 'Chelsea', 500]])
df.columns = ['Organization', 'Salesperson Name', 'Sales']
print(df)

# group column Organization
print(df.groupby('Organization'))
# after it created, we can call operations
print(df.groupby('Organization').mean())
print(df.groupby('Organization').sum())
print(df.groupby('Organization').count())

# concat method
df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                    'B': ['B0', 'B1', 'B2', 'B3'],
                    'C': ['C0', 'C1', 'C2', 'C3'],
                    'D': ['D0', 'D1', 'D2', 'D3']},
                   index=[0, 1, 2, 3])

df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],
                    'B': ['B4', 'B5', 'B6', 'B7'],
                    'C': ['C4', 'C5', 'C6', 'C7'],
                    'D': ['D4', 'D5', 'D6', 'D7']},
                   index=[4, 5, 6, 7])

df3 = pd.DataFrame({'A': ['A8', 'A9', 'A10', 'A11'],
                    'B': ['B8', 'B9', 'B10', 'B11'],
                    'C': ['C8', 'C9', 'C10', 'C11'],
                    'D': ['D8', 'D9', 'D10', 'D11']},
                   index=[8, 9, 10, 11])

# DataFrame's concatenation by concat method
print(pd.concat([df1, df2, df3]))
# concat by column
print(pd.concat([df1, df2, df3], axis=1))

# merge method
leftDataFrame = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                              'A': ['A0', 'A1', 'A2', 'A3'],
                              'B': ['B0', 'B1', 'B2', 'B3']})

rightDataFrame = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                               'C': ['C0', 'C1', 'C2', 'C3'],
                               'D': ['D0', 'D1', 'D2', 'D3']})

print(pd.merge(leftDataFrame, rightDataFrame, how='inner', on='key'))

# join method
leftDataFrame = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                              'B': ['B0', 'B1', 'B2', 'B3']},
                             index=['K0', 'K1', 'K2', 'K3'])

rightDataFrame = pd.DataFrame({'C': ['C0', 'C1', 'C2', 'C3'],
                               'D': ['D0', 'D1', 'D2', 'D3']},
                              index=['K0', 'K1', 'K2', 'K3'])

df = leftDataFrame.join(rightDataFrame)
print(df)

# unique method
print(df['A'].unique())

# nunique method
print(df['A'].nunique())

# count value
print(df['A'].value_counts())

# sort value on column B
print(dataframe1)
print(dataframe1.sort_values('B'))

# read csv file
new_data_frame = pd.read_csv('stock_prices.csv')
print(new_data_frame)

# export csv file using pandas
dataframe1.to_csv('my_new_csv.csv')
# to remove the blank index column
dataframe1.to_csv('my_new_csv.csv', index=False)

# we can use the same method to import and export file
# such as .xlsx and .json (read_xlsx(), read_json()

# we can also pass the file url into pd.read_csv('https://...')