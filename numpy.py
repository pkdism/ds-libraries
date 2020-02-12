import numpy as np


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


# PRINTING ARRAYS
print(np.arange(6))
print(np.arange(12).reshape(4, 3))
print(np.arange(24).reshape(2, 3, 4))
print(np.arange(10000))
print(np.arange(10000).reshape(100, 100))
#np.set_printoptions(threshold = sys.maxsize)


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
b = np.arange(12).reshape(2, 2, 3)
b
b.sum(axis = 1)















