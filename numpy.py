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





























