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



# missing values
df1 = df.reindex(index = dates[0:4], columns = list(df.columns) + ['E'])
df1

df1.loc[dates[0]:dates[1], 'E'] = 1
df1

df1.dropna(how = 'any')

df1.fillna(value = 5)

pd.isna(df1)


# Operations
# stats
df

df.mean(axis = 0)
df.mean(axis = 1)


# apply
df.apply(np.cumsum)
df.apply(lambda x: x + 2)
df.apply(lambda x: x.max())


# histogramming
s = pd.Series(np.random.randint(0, 7, size = 10))
s.value_counts()


# string methods
s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])
s.str.lower()

df = pd.DataFrame(np.random.randn(10, 4))

pieces = [df[:3], df[3:7], df[7:]]

pd.concat(pieces)


# joins
left = pd.DataFrame({'key': ['foo', 'foo'], 'lval': [1, 2]})

right = pd.DataFrame({'key': ['foo', 'foo'], 'rval': [4, 5]})

pd.merge(left, right, on = 'key')

left = pd.DataFrame({'key': ['foo', 'bar'], 'lval': [1, 2]})

right = pd.DataFrame({'key': ['foo', 'bar'], 'rval': [4, 5]})

pd.merge(left, right, on = 'key')



df = pd.DataFrame({'A': ['foo', 'bar', 'foo', 'bar',
                      'foo', 'bar', 'foo', 'foo'],
                'B': ['one', 'one', 'two', 'three',
                      'two', 'two', 'one', 'three'],
                'C': np.random.randn(8),
                'D': np.random.randn(8)})

df.groupby('A').sum()
df.groupby(['A', 'B']).sum()


# Stack
tuples = list(zip(*[['bar', 'bar', 'baz', 'baz',
                      'foo', 'foo', 'qux', 'qux'],
                     ['one', 'two', 'one', 'two',
                      'one', 'two', 'one', 'two']]))

index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])

df = pd.DataFrame(np.random.randn(8, 2), index = index, columns = ['A', 'B'])

df2 = df[:4]

stacked = df2.stack()
unstacked = stacked.unstack()


# pivot tables
df = pd.DataFrame({'A': ['one', 'one', 'two', 'three'] * 3,
                    'B': ['A', 'B', 'C'] * 4,
                    'C': ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 2,
                    'D': np.random.randn(12),
                    'E': np.random.randn(12)})


pd.pivot_table(df, values = 'D', index = ['A', 'B'], columns = 'C')



# Time Series
rng = pd.date_range('1/25/2012', periods = 100, freq = 'S') # seconds

pd.date_range('1/25/2012', periods = 10, freq = 'T') # minutes
pd.date_range('1/25/2012', periods = 10, freq = '5T') # 5 minutes


ts = pd.Series(np.random.randint(0, 500, len(rng)), index = rng)

ts.resample('5Min').sum()


# time zone
rng = pd.date_range('3/6/2012 00:00', periods=5, freq='D')
ts = pd.Series(np.random.randn(len(rng)), rng)
ts_utc = ts.tz_localize('UTC')
ts_utc.tz_convert('US/Eastern')


rng = pd.date_range('1/1/2012', periods=5, freq='M')
ts = pd.Series(np.random.randn(len(rng)), index=rng)
ts
ps = ts.to_period()
ps.to_timestamp()



# Categoricals
df = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6],
                   "raw_grade": ['a', 'b', 'b', 'a', 'a', 'e']})
df.dtypes

df["grade"] = df["raw_grade"].astype("category")
df.dtypes

df["grade"]

df["grade"].cat.categories = ["very good", "good", "very bad"]

df["grade"] = df["grade"].cat.set_categories(["very bad", "bad", "medium",
                                               "good", "very good"])


df.sort_values(by = "grade") # sorting by category order, not lexicographically

df.groupby("grade").size() # grouping by categorical column also shows empty categories



# Plotting
import matplotlib.pyplot as plt
plt.close('all')

ts = pd.Series(np.random.randn(1000), index = pd.date_range('1/1/2000', periods = 1000))
ts = ts.cumsum()
ts.plot()


df = pd.DataFrame(np.random.randn(1000, 4), index = ts.index, columns = ['A', 'B', 'C', 'D'])

df = df.cumsum()

plt.figure()

df.plot()

plt.legend(loc = 'best')



# Getting data in and out of

# csv

df.to_csv('foo.csv')
pd.read_csv('foo.csv')


# hdf5
df.to_hdf('foo.h5', 'df')
pd.read_hdf('foo.h5', 'df')


# excel
df.to_excel('foo.xlsx', sheet_name = 'Sheet1')
pd.read_excel('foo.xlsx', 'Sheet1', index_col = None, na_values = ['NA'])



# gotchas
# if pd.Series([True, False, True]):
#     print('I was true')

if pd.Series([True, False, True]).any():
    print('I was true')