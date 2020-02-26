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



# -----------------------------------------------------------------------------
# SPECIAL FUNCTIONS

help(special)

# Bessel functions of real order (jv, j_zeros)
from scipy import special

special.jn_zeros(10, 2)
special.jn_zeros(10, 5)

theta = np.r_[0:2*np.pi:50j] # 50 values between 0 and 2*pi
radius = np.r_[0:1:50j] # 50 values between 0 and 1

np.cos(theta)
np.sin(theta)

x = np.array([r * np.cos(theta) for r in radius])
x.shape
x.size

# Cython bindings for special functions
%load_ext cython

%%cython
cimport scipy.special.cython_special as csc

cdef:
    double x = 1
    double complex z = 1 + 1j
    double si, ci, rgam
    double complex cgam

rgam = csc.gamma(x)
print(rgam)
cgam = csc.gamma(z)
print(cgam)
csc.sici(x, &si, &ci)
print(si, ci)


# Avoiding python function overhead
# Same functions from cpython implementation are faster
import scipy.special as sc
%%cython
cimport scipy.special.cython_special as csc

def python_tight_loop():
    cdef:
        int n
        double x = 1

    for n in range(100):
        sc.jv(n, x)

        
def cython_tight_loop():
    cdef:
        int n
        double x = 1

    for n in range(100):
        csc.jv(n, x)
        

# Special functions can be evaluated in parallel using cython bindings


# Functions not in scipy.special
def binary_entropy(x):
    return -(sc.xlogy(x, x) + sc.xlog1py(1 - x, -x))/np.log(2)

def step(x):
    return 0.5 * (np.sign(x) + np.sign(1 - x))

def ramp(x):
    return np.maximum(0, x)



# -----------------------------------------------------------------------------
# INTEGRATION

# General integration (quad)
import scipy.integrate as integrate
import scipy.special as special
result = integrate.quad(lambda x: special.jv(2.5, x), 0, 4.5) # integration of bessel function jv(2.5, x) along [0, 4.5]
result

from numpy import sqrt, sin, cos, pi
I = sqrt(2/pi)*(18.0/27*sqrt(2)*cos(4.5) - 4.0/27*sqrt(2)*sin(4.5) + sqrt(2*pi) * special.fresnel(3/sqrt(pi))[0])
I
print(abs(result[0] - I))

from scipy.integrate import quad

def integrand(x, a, b):
    return a*x**2 + b

a = 2
b = 1

I = quad(integrand, 0, 1, args = (a, b))
I


from scipy.integrate import quad
def integrand(t, n, x):
    return np.exp(-x*t) / t**n

def expint(n, x):
    return quad(integrand, 1, np.inf, args = (n, x))[0]

vec_expint = np.vectorize(expint)
vec_expint(3, np.arange(1.0, 4.0, 0.5))

import scipy.special as special
special.expn(3, np.arange(1.0, 4.0, 0.5))

result = quad(lambda x: expint(3, x), 0, np.inf)
result

I3 = 1.0/3.0
print(I3)

print(I3 - result[0])


# General multiple integration (dblquad, tplquad, nquad)

from scipy.integrate import quad, dblquad

def I(n):
    return dblquad(lambda t, x: np.exp(-x*t)/t**n, 0, np.inf, lambda x: 1, lambda x: np.inf)

print(I(4))
print(I(3))
print(I(2))



area = dblquad(lambda x, y: x*y, 0, 0.5, lambda x: 0, lambda x: 1-2*x)
area

from scipy import integrate
N = 5
def f(t, x):
    return np.exp(-x*t) / t**N

integrate.nquad(f, [[1, np.inf], [0, np.inf]])


from scipy import integrate
def f(x, y):
    return x * y

def bounds_y():
    return [0, 0.5]

def bounds_x(y):
    return [0, 1-2*y]

integrate.nquad(f, [bounds_x, bounds_y])



# -----------------------------------------------------------------------------
# OPTIMIZATION AND INTERPOLATION
from scipy import optimize, interpolate
help(optimize)
help(interpolate)



# -----------------------------------------------------------------------------
# FOURIER TRANSFORMS

from scipy import fft, ifft
x = np.array([1, 2, 1, -1, 1.5])
y = fft(x)
yinv = ifft(y)
yinv
np.sum(x)

N = 600
T = 1/800
x = np.linspace(0, N*T, N)
y = np.sin(50*2*np.pi*x) + 0.5*np.sin(80*2*np.pi*x)
yf = fft(y)
xf = np.linspace(0, 1/(2*T), N//2)

import matplotlib.pyplot as plt
plt.plot(xf, 2/N*np.abs(yf[0:N//2]))
plt.grid()
plt.show()



# -----------------------------------------------------------------------------
# SIGNAL PROCESSING

import numpy as np
from scipy import signal, misc
import matplotlib.pyplot as plt
help(signal)

image = misc.face(gray = True).astype(np.float32)
derfilt = np.array([1.0, -2, 1.0], dtype = np.float32)
ck = signal.cspline2d(image, 8.0)
deriv = (signal.sepfir2d(ck, derfilt, [1]) + signal.sepfir2d(ck, [1], derfilt))
plt.figure()
plt.imshow(image)
plt.gray()
plt.title('Original image')
plt.show()

plt.figure()
plt.imshow(deriv)
plt.gray()
plt.title('Output of spline edge filter')
plt.show()

image = misc.face(gray = True)
w = np.zeros((50, 50))
w[0][0] = 1.0
w[49][25] = 1.0
image_new = signal.fftconvolve(image, w)

plt.figure()
plt.imshow(image)
plt.gray()
plt.title('Original image')
plt.show()

plt.figure()
plt.imshow(image_new)
plt.gray()
plt.title('Filtered image')
plt.show()


image = misc.ascent()
w = signal.gaussian(50, 10.0)
image_new = signal.sepfir2d(image, w, w)

plt.figure()
plt.imshow(image)
plt.gray()
plt.title('Original image')
plt.show()

plt.figure()
plt.imshow(image_new)
plt.gray()
plt.title('Filtered image')
plt.show()