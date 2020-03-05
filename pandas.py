import numpy as np
import pandas as pd


# Series
s = pd.Series([1, 3, 5, np.nan, 6, 8])
s.dtype

a = np.array([1, 3, 4, np.nan, 6, 8])
a.dtype

t = pd.Series([1, 3, 5, np.nan, 'a', "xyz"])
t.dtype

t[0] + 2
# t[4] + 3 # error
t[4] + 'w'


# Data frames
dates = pd.date_range('20100101', periods = 6)

df = pd.DataFrame(np.random.randn(6, 4), index = dates, columns = list('ABCD'))

df2 = pd.DataFrame({'A': 1.,
                    'B': pd.Timestamp('20130102'),
                    'C': pd.Series(1, index = list(range(4))),
                    'D': np.array([3]*4),
                    'E': pd.Categorical(['test', 'train', 'test', 'train']),
                    'F': 'foo'})
df2
df2.dtypes

# Viewing data
df.head()
df.tail(3)
df.index
df.columns

df.to_numpy() # all casted to float
df2.to_numpy() # all casted to object

df.describe()
df2.describe() # only describes the numeric ones

df.T

df.sort_index(axis = 1, ascending = False) # sort by columns (axis = 1)
df.sort_index(axis = 0, ascending = False) # sort by rows (axis = 0)

df.sort_values(by = 'B') # sort by values


# Selection
df['A']
df.A

df[0:3] # excludes last index
df['20100102':'20100104'] # includes last index

df.loc[dates[0]]
df.loc[:, ['A', 'B']]

df.loc['20100102':'20100104', ['A', 'B']]

df.loc['20100102', ['A', 'B']]

df.loc[dates[0], 'A']

df.iloc[3]

df.iloc[3, :]
df.iloc[:, 3]
df.loc[:, 'D']

df.iloc[3:5, 0:2]

df.iloc[[1, 2, 4], [0, 2]]

df.iloc[1:3, :]
df.iloc[:, 1:3]

df.iloc[1, 1]
df.iat[1, 1]


# Boolean columns and indexes
df[df['A'] > 0]

df[df > 0]

# .isin() - same as %in% in R
df2 = df.copy()

df2['E'] = ['one', 'one', 'two', 'three', 'four', 'three']

df2[df2['E'].isin(['two', 'three'])]

# Setting

s1 = pd.Series([1, 2, 3, 4, 5, 6], index = pd.date_range('20100102', periods = 6))

s1

df['F'] = s1

df.at[dates[0], 'A'] = 0
df.iat[0, 1] = 0

df.loc[:, 'D'] = np.array([5]*len(df))

df


# where operation
df2 = df.copy()

df2[df2 > 0] = -df2










































