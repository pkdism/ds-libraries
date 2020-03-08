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

























