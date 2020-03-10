import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


fig, ax = plt.subplots()
ax.plot([1, 2, 3, 4], [1, 4, 2, 3])

plt.plot([1, 2, 3, 4], [1, 4, 2, 3])

fig = plt.figure()
fig, ax = plt.subplots()
fig, axs = plt.subplots(2, 2)

# only numpy.array objects will always give the desired result, other like data types may or may not
a = pd.DataFrame(np.random.rand(4, 5), columns = list('abcde'))
a_asarray = a.values # converting pandas dataframe to a numpy.array

b = np.matrix([[1, 2], [3, 4]])
b_asarray = np.asarray(b)


# the object-oriented interface to use matplotlib
x = np.linspace(0, 2, 100)
fig, ax = plt.subplots()
ax.plot(x, x, label = 'linear')
ax.plot(x, x ** 2, label = 'quadratic')
ax.plot(x, x ** 3, label = 'cubic')
ax.set_xlabel('x label')
ax.set_ylabel('y label')
ax.set_title("Simple Plot")
ax.legend()

# pyplot style interface to use matplotlib
x = np.linspace(0, 2, 100)
plt.plot(x, x, label = 'linear')
plt.plot(x, x ** 2, label = 'quadratic')
plt.plot(x, x ** 3, label = 'cubic')
plt.xlabel('x label')
plt.ylabel('y label')
plt.title('Simple Plot')
plt.legend()


# function signature to draw same plot with different datasets
def my_plotter(ax, data1, data2, param_dict):
    """
    A helper function to make a graph

    Parameters
    ----------
    ax : Axes
        The axes to draw to

    data1 : array
       The x data

    data2 : array
       The y data

    param_dict : dict
       Dictionary of kwargs to pass to ax.plot

    Returns
    -------
    out : list
        list of artists added
    """
    out = ax.plot(data1, data2, **param_dict)
    return out


data1, data2, data3, data4 = np.random.randn(4, 100)
fig, ax = plt.subplots()
my_plotter(ax, data1, data2, {'marker': 'x'})

fig, (ax1, ax2) = plt.subplots(1, 2)
my_plotter(ax1, data1, data2, {'marker': 'x'})
my_plotter(ax2, data3, data4, {'marker': 'o'})



# PYPLOT

plt.plot([1,2,3,4]) # in case of single array, values are y values and x are taken by default - starting from 0
plt.ylabel('some numbers')
plt.show()

plt.plot([1, 2, 3, 4], [1, 4, 9, 16]) # first array has x values and second has y values

plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro') # ro stands for red o's or circles
plt.axis([0, 6, 0, 20]) # sets xlim and ylim
plt.show()


# evenly sampled time at 200ms intervals
t = np.arange(0., 5., 0.2)

# red dashes, blue squares and green triangles
plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^') # r-- red dashes, bs - blue squares, g^ green triangles
plt.show()


data = {'a': np.arange(50),
        'c': np.random.randint(0, 50, 50),
        'd': np.random.randn(50)}
data['b'] = data['a'] + 10 * np.random.randn(50)
data['d'] = np.abs(data['d']) * 100

plt.scatter('a', 'b', c='c', s='d', data=data) # c stands for color and s stands for size
plt.xlabel('entry a')
plt.ylabel('entry b')
plt.show()


# Categorical variables
names = ['group_a', 'group_b', 'group_c']
values = [1, 10, 100]

plt.figure(figsize=(9, 3))
plt.subplot(131) # number of rows = 1, number of columns = 3, plot number = 1
plt.bar(names, values)
plt.subplot(132) # number of rows = 1, number of columns = 3, plot number = 2
plt.scatter(names, values)
plt.subplot(133) # number of rows = 1, number of columns = 3, plot number = 3
plt.plot(names, values)
plt.suptitle('Categorical Plotting')
plt.show()
plt.show()


x = list(range(20))
y = [i ** 2 for i in x]
plt.plot(x, y, linewidth=2.0)


line, = plt.plot(x, y, '-')
line.set_antialiased(False) # turn off antialiasing

# Creating multiple figures and axes
def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)

t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)

plt.figure() # optional
plt.subplot(211) # number of rows = 2, number of columns = 1, plot number = 1
plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')

plt.subplot(212) # number of rows = 1, number of columns = 3, plot number = 2
plt.plot(t2, np.cos(2*np.pi*t2), 'r--')
plt.show()


plt.figure(1)                # the first figure
plt.subplot(211)             # the first subplot in the first figure
plt.plot([1, 2, 3])
plt.subplot(212)             # the second subplot in the first figure
plt.plot([4, 5, 6])

plt.figure(2)                # a second figure
plt.plot([4, 5, 6])          # creates a subplot(111) by default

plt.figure(1)                # figure 1 current; subplot(212) still current
plt.subplot(211)             # make subplot(211) in figure1 current
plt.title('Easy as 1, 2, 3') # subplot 211 title

# cla() # clears current axes
# clf() # clears currect figure


# Working with texts
mu, sigma = 100, 15
x = mu + sigma * np.random.randn(10000)

# the histogram of the data
n, bins, patches = plt.hist(x, 50, density=1, facecolor='g', alpha=0.75)
plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title('Histogram of IQ')
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.axis([40, 160, 0, 0.03])
plt.grid(True)
plt.show()

t = plt.xlabel('my data', fontsize=14, color='red')
plt.title(r'$\sigma_i=15$') # using LaTeX notations


# Annotations
ax = plt.subplot(111)
t = np.arange(0.0, 5.0, 0.01)
s = np.cos(2*np.pi*t)
line, = plt.plot(t, s, lw=2)
plt.annotate('local max', xy=(2, 1), xytext=(3, 1.5), 
             arrowprops=dict(facecolor='black', shrink=0.05),
             ) # xy arguement is part of the plot to point to and xytext is place of text

plt.ylim(-2, 2)
plt.show()


# Logarithmic and non-linear axes

# Fixing random state for reproducibility
np.random.seed(19680801)

# make up some data in the open interval (0, 1)
y = np.random.normal(loc=0.5, scale=0.4, size=1000)
y = y[(y > 0) & (y < 1)]
y.sort()
x = np.arange(len(y))

# plot with various axes scales
plt.figure()

# linear
plt.subplot(221)
plt.plot(x, y)
plt.yscale('linear')
plt.title('linear')
plt.grid(True)


# log
plt.subplot(222)
plt.plot(x, y)
plt.yscale('log')
plt.title('log')
plt.grid(True)


# symmetric log
plt.subplot(223)
plt.plot(x, y - y.mean())
plt.yscale('symlog', linthreshy=0.01)
plt.title('symlog')
plt.grid(True)

# logit
plt.subplot(224)
plt.plot(x, y)
plt.yscale('logit')
plt.title('logit')
plt.grid(True)
# Adjust the subplot layout, because the logit one may take more space
# than usual, due to y-tick labels like "1 - 10^{-3}"
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)

plt.show()



# IMAGES

# only .png images are supported
import matplotlib.image as mpimg

# getting the image into a numpy array with 4 dimensions - red, green, blue, alpha (transperency)
img = mpimg.imread('/Users/pawan1.dwivedi/Downloads/pics/minimalist/6005397-minimalist-wallpaper.png')
print(img)
img.shape
img.dtype

# plotting a numpy array
imgplot = plt.imshow(img)


# picking one channel of numpy
lum_img = img[:, :, 0]
plt.imshow(lum_img)

plt.imshow(lum_img, cmap="hot")


