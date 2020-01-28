# A basic example

import numpy as np
a = np.arange(15).reshape(5, 3)

a.shape

a.ndim

a.dtype

a.itemsize

a.size

type(a)

b = np.array([6, 7, 8])

b

type(b)

# Array creation
a = np.array([2, 3, 4])

a

a.dtype

b = np.array([1.2, 3.5, 5.1])

b.dtype

b = np.array([(1.5, 2, 3), (4, 5, 6)])

b

c = np.array([ [1, 2], [3, 4] ], dtype = complex)

c

np.zeros((3, 4))

np.ones((2, 3, 4), dtype = np.int16)

np.arange(10, 30, 5)

np.arange(0, 2, 0.3)

from numpy import pi

np.linspace(0, 2, 9)

x = np.linspace(0, 2*pi, 100)

f = np.sin(x)





