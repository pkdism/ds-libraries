import numpy as np

# -----------------------------------------------------------------------------
# AN EXAMPLE
a = np.arange(15).reshape(3,5)
a
a.shape
a.ndim
a.dtype.name
a.itemsize
a.size
type(a)

b = np.array([6, 7, 8])
b
type(b)



# -----------------------------------------------------------------------------
# ARRAY CREATION
a = np.array([2, 3, 4])
a
a.dtype

b = np.array([1.2, 3.5, 5.1])
b
b.dtype

b = np.array([ [1.5, 2, 3], (4, 5, 6)])
b

c = np.array([ [1, 2], [3, 4] ], dtype = complex)
c


# zeros, ones and empty
np.zeros((3, 4))
np.ones((2, 3, 4), dtype = np.int16)
np.empty((2, 3))
x = np.arange(6).reshape((2, 3))
x
np.zeros_like(x)
np.ones_like(x)
np.empty_like(x)


# from function and from file
np.fromfunction(lambda i, j: i == j, (3, 3), dtype = int)
np.fromfunction(lambda i, j: i + j, (3, 3), dtype = int)
# np.fromfile(fname, dtype)


# arange and linspace
np.arange(10, 30, 5)
np.arange(0, 2, 0.3)
np.arange(10)
np.arange(10, 30)

from numpy import pi
np.linspace(0, 2, 9) # 9 numbers from 0 to 2
x = np.linspace(0, 2*pi, 100) # useful to evaluate a lot of points
f = np.sin(x)



# -----------------------------------------------------------------------------
# PRINTING ARRAYS
print(np.arange(6))
print(np.arange(12).reshape(4, 3))
print(np.arange(24).reshape(2, 3, 4))
print(np.arange(10000))
print(np.arange(10000).reshape(100, 100))
#np.set_printoptions(threshold = sys.maxsize)



# -----------------------------------------------------------------------------
# BASIC OPERATIONS
a = np.array([20, 30, 40, 50])
b = np.arange(4)
b
c = a - b # arithmetic operators or arrays apply element-wise. A new array is created and filled with result.
c
b ** 2
10 * np.sin(a)
a < 35

# matrix multiplication
A = np.array([[1, 1],
              [0, 1]])
B = np.array([[2, 0],
              [3, 4]])
A * B # element wise multiplication
A @ B # dot product
A.dot(B) # dot product

# += and *=
a = np.ones((2, 3), dtype = int)
b = np.random.rand(2, 3)
a += 3 # acts in place - without creating a new array
b += a
a += b # b is not automatically converted into integer type because float is more precise than integer

# upcasting
a = np.ones(3, dtype = np.int32)
b = np.linspace(0, pi, 3)
b.dtype.name
c = a + b
c.dtype.name
d = np.exp(c*1j)
d
d.dtype.name

# sum, min, max
a = np.random.rand(2, 3)
a
a.sum()
a.min()
a.max()

# sum, min, max on specific axis
b = np.arange(12).reshape(3, 4)
b
b.sum(axis = 1)
b.sum(axis = 0)

b.cumsum(axis = 1)



# -----------------------------------------------------------------------------
# UNIVERSAL FUNCTIONS
B = np.arange(3)
B
np.exp(B)
np.sqrt(B)
np.sin(B)

C = np.array([0, 1, 1.4142])
np.add(B, C)

# all, any, apply_along_axis, argmax, argmin, argsort, average, bincount, ceil,
# clip, conj, corrcoef, cov, cross, cumprod, cumsum, diff, dot, floor, inner, 
# invert, lexsort, max, maximum, mean, median, min, minimum, nonzero, outer, 
# prod, re, round, sort, std, sum, trace, transpose, var, vdot, vectorize, where



# -----------------------------------------------------------------------------
# INDEXING, SLICING AND ITERATING

# one dimensional arrays
a = np.arange(10)**3
a
a[2]
a[2:5] # a[5] is not included
a[0:6:2] = 1000 # from 0 to 6, exclusive, set every second element to 1000
a
a[:6:2] = 1000
a
a[::-1] # reversed a
for i in a:
    print(i**(1/3))

# multidimensional arrays
def f(x, y):
    return 10*x+y

b = np.fromfunction(f, (5, 4), dtype = int)
b
b[2,3]
b[0:5, 1]
b[:, 1]
b[1:3, :]
b[-1]
b[-1,:]
b[-1,...]
x = np.arange(720).reshape(6, 5, 4, 3, 2)
x[1, 2, ...]
x[1, 2, :, :, :]
x[1, 2, ...] == x[1, 2, :, :, :]
sum(x[1, 2, ...]) == sum(x[1, 2, :, :, :])
sum(sum(x[1, 2, ...])) == sum(sum(x[1, 2, :, :, :]))
sum(sum(sum(x[1, 2, ...]))) == sum(sum(sum(x[1, 2, :, :, :])))

x[..., 1]
x[:, :, :, :, 1]
x[4, ..., 2, :]
x[4, :, :, 2, :]

c = np.array([[[0, 1, 2],
               [10, 12, 13]],
            [[100, 101, 102],
             [110, 112, 113]]])
c.shape
c[1, ...]
c[..., 2]
c[:, :, 2]
c[:, 1, :]

for row in b:
    print(row)

for element in b.flat:
    print(element)



# -----------------------------------------------------------------------------
# SHAPE MANIPULATION
rg = np.random.rand
a = np.floor(10*rg(3, 4))
a.shape
a.ravel()
a.reshape(6, 2) # reshape will return a reshaped array, w/o modifying the array itself
a.T
a.T.shape
a.shape
a
a.resize((2, 6)) # resize modifies the array itself
a
a.reshape(3, -1) # if a dimension is given as -1, other dimensions are calculated automatically



# -----------------------------------------------------------------------------
# STACKING TOGETHER DIFFERENT ARRAYS

a = np.floor(10*rg(2, 2))
a
b = np.floor(10*rg(2, 2))
b
np.vstack((a, b))
np.hstack((a, b))

from numpy import newaxis
np.column_stack((a, b))
np.hstack((a, b))

a = np.array((4, 2))
b = np.array((3, 8))
np.column_stack((a, b))
np.hstack((a, b))

a[:, newaxis] # 2-d columns vector

a[:, newaxis][0][0]
a[:, newaxis][1][0]

np.column_stack((a, b))
np.column_stack((a[:, newaxis], b[:, newaxis]))
np.hstack((a[:, newaxis], b[:, newaxis]))


np.column_stack is np.hstack
np.row_stack is np.vstack

np.r_[1:4, 0, 4]
np.c_[1:4]

np.c_[np.array([[1, 2, 3]]), np.array([0]), np.array([4])]



# -----------------------------------------------------------------------------
# SPLITTING ONE ARRAY INTO SEVERAL SMALLER ONES

a = np.floor(10*rg(2, 12))
a
np.hsplit(a, 3)
np.hsplit(a, (3, 4))



# -----------------------------------------------------------------------------
# COPIES AND VIEWS

# 1. No copy at all
a = np.arange(12)
b = a # no new object is created
b is a # same ndarray object has 2 names
b.shape = 3,4 # a is reshaped
a.shape


# Python passes mutable object as references
def f(x):
    print(id(x))

id(a)
f(a)


# 2. View or shallow copy
# Different array objects can share the same data. 
# The view method creates a new array that looks at the same data.
c = a.view()
c is a
c.base is a # c is the view of data owned by a
c.flags.owndata
c.shape = 2, 6
a.shape
c[0, 4] = 1234 # a's data changes
a


# Slicing an array returns a view of it.
s = a[:, 1:3]
s[:] = 10 # s[:] is a view of s
a


# 3. Deep copy
# The copy method makes a complete copy of the array and its data
d = a.copy()
d is a
d.base is a
d[0, 0] = 999
a

a = np.arange(int(1e8))
b = a[:100].copy()
del a # the memory of 'a' can be released



# -----------------------------------------------------------------------------
# FANCY INDEXING AND INDEX TRICKS

# Indexing with arrays of indices
a = np.arange(12) ** 2
i = np.array([1, 1, 3, 8, 5]) # an array of indices
a[i]

j = np.array([[3, 4], [9, 7]]) # a bi-dimensional array of indices
a[j]


# When the indexed array a is multidimensional, 
# a single array of indices refers to the first dimension of a. 
# The following example shows this behavior by converting an image of labels 
# into a color image using a palette.

palette = np.array([[  0,   0,   0],  # black
                    [255,   0,   0],  # red
                    [  0, 255,   0],  # green
                    [  0,   0, 255],  # blue
                    [255, 255, 255]]) # white


image = np.array([ [0, 1, 2, 0],
                   [0, 3, 4, 0] ])

palette[image] # the 2, 3, 4 color image




# We can also give indices for multiple dimensions
a = np.arange(12).reshape(3, 4)
a
i = np.array([ [0, 1], # indices for first dimension
               [1, 2] ])
j = np.array([ [0, 1], # indices for second dimension
               [1, 2] ])
a[i, j]
j = np.array([ [2, 1],
               [3, 3] ])
a[i, j]
a[i, 2]
a[:, j]

l = (i, j)
a[l] # equivalent to a[i, j]


s = np.array([i, j])
a[s] # not what we want
a[tuple(s)] # same as a[i, j]


time = np.linspace(20, 145, 5) # time scale
data = np.sin(np.arange(20).reshape(5, 4)) # 4 time-dependent series

# index of maxima of each series
ind = data.argmax(axis = 0)
ind

# time corresponding to the maxima
time_max = time[ind]

data_max = data[ind, range(data.shape[1])] # => data[ind[0], 0], data[ind[1], 1], ...

time_max
data_max

np.all(data_max == data.max(axis = 0))

# you can also use indexing with arrays as a target to assign to
a = np.arange(5)
a

a[[1, 3, 4]] = 0
a

# when the list of indices contains repetitions, the assignment is done several times, leaving behind the last value
a = np.arange(5)
a[[0, 0, 2]] = [1, 2, 3]
a

# += -> even if 0 occurs twice, the 0th element in only incremented once
a = np.arange(5)
a
a[[0, 0, 2]] += 1 # this is same as a[[0, 0, 2]] = a[[0, 0, 2]] + 1



# -----------------------------------------------------------------------------
# INDEXING WITH BOOLEAN ARRAYS

a = np.arange(12).reshape(3, 4)
a
b = a > 4 # boolean array - same shape as a
b
a[b] # a 1-D array with selected elements

# this can be very useful in assignment
a[b] = 0
a

# second way of indexing with boolean arrays
# for each dimension, give a 1-D array
a = np.arange(12).reshape(3, 4)
a
b1 = np.array([False, True, True]) # first dimension selection - length of b1 should be equal to nrows of a
b2 = np.array([True, False, True, False]) # second dimesion selection - length of b2 should be equal to ncols of a

a[b1, :] # selecting rows
a[b1] # same thing

a[:, b2] # selecting columns

a[b1, b2]



# -----------------------------------------------------------------------------
# ix_() function

a = np.array([2, 3, 4, 5])
b = np.array([8, 5, 4])
c = np.array([5, 4, 6, 8, 3])

ax, bx, cx = np.ix_(a, b, c)
ax
bx
cx
ax.shape
bx.shape
cx.shape

result = ax + bx * cx
result

result[3, 2, 4]
result[3, 2]

a[3] + b[2] * c[4]






