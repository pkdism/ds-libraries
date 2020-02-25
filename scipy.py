# -----------------------------------------------------------------------------
# INTRODUCTION

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy import linalg, optimize
print(np.info(optimize.fmin))



# -----------------------------------------------------------------------------
# BASIC FUNCTIONS

# Index tricks
a = np.r_[3, [0]*5, np.arange(-1, 1.002, 2/9.0)]
a
a = np.r_[3, [0]*5, -1:1:10j]
a
np.mgrid[0:5, 0:5]
np.mgrid[0:5:4j, 0:5:4j]

# Polynomials
from numpy import poly1d
p = poly1d([3, 4, 5])
print(p)
# print(p*p)
print(p.integ(k = 6)) # k is constant of indefinite integral
print(p.deriv())
print(p.deriv(2)) # second derivative
p([4, 5])

# Vectorizing functions (vectorize)
def addsubtract(a, b):
    if a > b:
        return a - b
    else:
        return a + b

vec_addsubtract = np.vectorize(addsubtract)
vec_addsubtract([0, 3, 6, 9], [1, 3, 5, 7])

# Type handling

a = np.array([1, 2, 3])
np.isreal(a)
np.isrealobj(a)

b = np.array([1 + 2j, 3 + 4j])
np.isreal(b)
np.iscomplex(b)
np.isrealobj(b)
np.iscomplexobj(b)

c = np.array([1 + 2j, 3])
np.isreal(c)
np.iscomplex(c)
np.isrealobj(c)
np.iscomplexobj(c)

# Other useful functions

np.linspace(1, 3, 10) # linear space
np.logspace(2, 3, 10) # log space

x = np.arange(10)
condlist = [x < 3, x > 5]
choicelist = [x, x ** 2]
np.select(condlist, choicelist)

from scipy import special
special.factorial(5) # n!
special.comb(3, 2) # nCr