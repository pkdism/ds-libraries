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



# -----------------------------------------------------------------------------
# LINEAR ALGEBRA
# scipy.linalg contains everything from numpy.linalg plus additional features
# Unless you want to avoid adding scipy dependency, always use scipy.linalg

import numpy as np
A = np.mat('[1, 2; 3, 4]') # numpy.matrix is more convenient for matrix operations than numpy.ndarray
A                          # despite its convenience, numpy.matrix is discouraged
A.I
A.T

b = np.mat('[5, 6]')
b
b.T
A*b.T

from scipy import linalg
A = np.array([[1, 2], [3, 4]])
A
linalg.inv(A)
A.T

b = np.array([[5, 6]]) # 2-D array
b.T

A*b # not matrix multiplication
A.dot(b.T) # matrix multiplication

b = np.array([5, 6]) # 1-D array
b
b.T # not matrix transpose

A.dot(b) # does not matter for multiplication


# Basic routines
# Finding the inverse
A = np.array([[1, 3, 5], [2, 5, 1], [2, 3, 8]])
linalg.inv(A)

A.dot(linalg.inv(A)) # Identity


# Solving a linear system
# x + 3y + 5z = 10
# 2x + 5y + z = 8
# 2x + 3y + 8z = 3

# Ax = b
# x = Ainv.b

A = np.array([[1, 3, 5], [2, 5, 1], [2, 3, 8]])
b = np.array([[10, 8, 3]])

Ainv = linalg.inv(A)

xyz1 = Ainv.dot(b.T) # slow, use linalg.solve instead

A.dot(xyz1) - b.T # check


xyz2 = linalg.solve(A, b.T)
A.dot(xyz2) - b.T # check


# Determinant
detA = linalg.det(A)


# Computing norms
A = np.array([[1, 2], [3, 4]])
A
linalg.norm(A)
linalg.norm(A, "f") # frobenius norm is the default
linalg.norm(A, 1) # L1 norm, max column sum
linalg.norm(A, np.inf) # L inf norm, max row sum
linalg.norm(A, -1) # min column sum
linalg.norm(A, -np.inf) # min row sum


# Solving linear least squares problem and pseudo inverses
c1, c2 = 5, 2
i = np.r_[1:11]
xi = 0.1*i
yi = c1*np.exp(-xi) + c2*xi
zi = yi + 0.05 * np.max(yi) * np.random.randn(len(yi))

A = np.c_[np.exp(-xi)[:, np.newaxis], xi[:, np.newaxis]]
c, resid, rand, sigma = linalg.lstsq(A, zi)

xi2 = np.r_[0.1:1.0:100j]
yi2 = c[0]*np.exp(-xi2) + c[1]*xi2

plt.plot(xi, zi, 'x', xi2, yi2)
plt.axis(0, 1.1, 3, 5.5)
plt.xlabel('$x_i$')
plt.title('Data fitting with linalg.lstsq')
plt.show()


# Generalized inverse
# Let A be M x N matrix
# A^† and A^# - Generalized inverse
# A^H - Hermitian transpose
# A^-1 - Inverse of matrix A
# Case 1: M > N then A^† = (A^H x A)^-1 x A^H
# Case 2: M < N then A^# = A^H x (A x A^H)^-1
# Case 3: M = N then A^† = A^# = A^-1 as long as A is invertible

# linalg.pinv uses linalg.lstsq to calculate generalized inverse
# linalg.pinv2 uses singular value decomposition to calculate generalized inverse


# DECOMPOSITION

## Eigenvalues and eigenvectors
### For some square matrix A, find vector v and scalar (lambda) ƛ such that Av = ƛv

### For a N x N matrix A igenvalues are roots of the polynomial |A - ƛI| = 0

### More general eigenvalue problem: Av = ƛBv
### The generalized solution is A = BVΛV^-1
### Λ is a diagonal matrix of eigon values
### V is a collection of eigenvectors into columns

A = np.array([[1, 5, 2], [2, 4, 1], [3, 6, 2]])
### |A - ƛI| = -ƛ^3 + 7ƛ^2 + 8ƛ - 3
### Roots are ƛ1 = 7.96, ƛ2 = -1.26, ƛ3 = 0.3. These are eigenvalues of A
### Eigenvectors corresponding to each eigenvalue can be found using the equation Aƛ = ƛv

la, v = linalg.eig(A)

A = np.array([[1, 2], [3, 4]])
la, v = linalg.eig(A)
l1, l2 = la
print(l1, l2) # eigenvalues
print(v[:, 0]) # first eigenvector
print(v[:, 1]) # second eigenvector

print(np.sum(abs(v**2), axis = 0)) # eigenvectors are unitary
v1 = np.array(v[:, 0]).T
print(linalg.norm(A.dot(v1) - l1*v1)) # check the computation


## Singular value decomposition

A = np.array([[1, 2, 3], [4, 5, 6]])
A
M, N = A.shape
U, s, Vh = linalg.svd(A)

Sig = linalg.diagsvd(s, M, N)
U, Vh = U, Vh
Sig

U.dot(Sig.dot(Vh)) # check computation
# A hermitian matrix D satisfies D^H = D
# A unitary matrix D satisfies D^HxD = I so that D^-1 = D^H


## LU Decomposition

# A = PLU where P is MxM permuation matrix (a permutation of rows of I)
# L is in MxK lower triangular (or trapezoidal) matrix with K = min(M, N) with unit diagonal
# and U is upper triangular or trapezoidal matrix

A
p, l, u = linalg.lu(A)


## Cholesky decomposition
# A = A^H and A = U^HxU or LL^H
# L = U^H


## QR decomposition
# A = QR, A: MxN, Q: MxM unitary matrix, R: MxN upper trapezoidal matrix



# MATRIX FUNCTIONS

# Exponential and logarithm functions - expm, logm
A_sq = np.array([[1, 2], [3, 4]])
A_non_sq = np.array([[1, 2, 3], [4, 5, 6]])

linalg.expm(A_sq)
linalg.logm(A_sq)

linalg.expm(A_non_sq)
linalg.logm(A_non_sq)

# Trigonometric functions - sinm, cosm, tanm
linalg.sinm(A_sq)
linalg.sinm(A_non_sq)

# Hyperbolic trigonometric functions - sinhm, coshm, tanhm
linalg.sinhm(A_sq)
linalg.sinhm(A_non_sq)

# Arbitrary function
from scipy import special, random, linalg
np.random.seed(1234)
A = random.rand(3, 3)
B = linalg.funm(A, lambda x: special.jv(0, x))
A
B
linalg.eigvals(A)
special.jv(0, linalg.eigvals(A))
linalg.eigvals(B)



# -----------------------------------------------------------------------------
# SPARSE EIGENVALUE PROBLEMS WITH ARPACK
# Can be used to find only smallest/largest/real/complex part eigenvalues

from scipy.linalg import eig, eigh
from scipy.sparse.linalg import eigs, eigsh

np.set_printoptions(suppress = True)

np.random.seed(0)

X = np.random.rand(100, 100) - 0.5

X = np.dot(X, X.T)

evals_all, evecs_all = eigh(X) # too many eigen values

evals_large, evecs_large = eigsh(X, 3, which = 'LM') # largest 3 eigenvalues

print(evals_all[-3:])
print(evals_large)



# -----------------------------------------------------------------------------
# SPATIAL DATA STRUCTURES AND ALGORITHMS

# Delaunay triangulations
from scipy.spatial import Delaunay
points = np.array([[0, 0], [0, 1.1], [1,0], [1,1], [2, 1], [1, 2], [2, 2], [1, 1.5], [1, 1.3]])
tri = Delaunay(points)

plt.triplot(points[:,0], points[:,1], tri.simplices)
plt.plot(points[:,0], points[:,1], 'o')
for j, p in enumerate(points):
    plt.text(p[0]-0.03, p[1]+0.03, j, ha='right') # label the points
for j, s in enumerate(tri.simplices):
    p = points[s].mean(axis=0)
    plt.text(p[0], p[1], '#%d' % j, ha='center') # label triangles
#plt.xlim(-0.5, 1.5); plt.ylim(-0.5, 1.5)
plt.show()


# Convex hulls
from scipy.spatial import ConvexHull, convex_hull_plot_2d
points = np.random.rand(30, 2)
hull = ConvexHull(points)
plt.plot(points[:, 0], points[:, 1], 'o')
for simplex in hull.simplices:
    plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
plt.show()

convex_hull_plot_2d(hull)



# -----------------------------------------------------------------------------
# STATS
from scipy import stats
help(stats)


# Random variables

from scipy.stats import norm # normal
# from __future__ import print_function

print(stats.norm.__doc__)

print('bounds lower %s, upper %s' %(norm.a, norm.b))

dir(norm)

rv = norm()
dir(rv)

dist_continu = [d for d in dir(stats) if isinstance(getattr(stats, d), stats.rv_continuous)]
dist_discrete = [d for d in dir(stats) if isinstance(getattr(stats, d), stats.rv_discrete)]



n = np.empty(1000)

for i in range(1000):
    n[i] = norm.rvs()

plt.hist(n, bins = 50, density = 1)
plt.hist(norm.pdf(n), bins = 50, density = 1)
plt.hist(norm.cdf(n), bins = 50, density = 1)
plt.hist(norm.moment(n.any()), bins = 50, density = 1)
norm.mean()
norm.var()
norm.ppf(0.5) # median of 0-1 normal distribution
norm.rvs(size = 3)

norm.rvs(size = 3, random_state = 1234)
norm.rvs(3) # the first arguement is the loc (mean, in case of normal distrubution) parameter, not size


# Shifting and scaling
x = norm.rvs(size = 5, loc = 3, scale = 4)
norm.stats(x)

norm.stats(loc = 3, scale = 4, moments = "mv")

np.mean(norm.rvs(5, size = 500))


# Shape parameters
from scipy.stats import gamma
gamma.numargs
gamma.shapes
gamma(a = 1, scale = 2).stats(moments = "mv")


# Freezing a distribution
rv = gamma(1, scale = 2) # first parameter is shape (alpha or a)
rv
rv.mean(), rv.std()

# continue after revising statistics