import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = 'darkgrid')

tips = sns.load_dataset("tips")


# VISUALIZING STATISTICAL RELATIONSHIPS

# Relating variables with scatter plots

# scatterplot
sns.relplot(x = "total_bill", y = "tip", data = tips)

# scatterplot with colored points
sns.relplot(x = "total_bill", y = "tip", hue = "smoker", data = tips)

# scatterplot with colored points and different marker style for each class
sns.relplot(x = "total_bill", y = "tip", hue = "smoker", style = "smoker", data = tips)

# use 4 variable - separate for hue and style
sns.relplot(x = "total_bill", y = "tip", hue = "smoker", style = "time", data = tips)

# using hue with numeric variable
sns.relplot(x = "total_bill", y = "tip", hue = "size", data = tips)

# customizing the color palette in case of numeric variables
sns.relplot(x = "total_bill", y = "tip", hue = "size", palette = "ch:r = -0.5, l = 0.75", data = tips)

# change the size of each point
sns.relplot(x = "total_bill", y = "tip", size = "size", data = tips)

# customize the size aesthetic for numeric variable
sns.relplot(x = "total_bill", y = "tip", size = "size", sizes = (15, 200), data = tips)


# Emphasizing continuity with line plots

df = pd.DataFrame(dict(time=np.arange(500),
                       value=np.random.randn(500).cumsum()))

g = sns.relplot(x="time", y="value", kind="line", data=df)
g.fig.autofmt_xdate()

df = pd.DataFrame(np.random.randn(500, 2).cumsum(axis=0), columns=["x", "y"])
sns.relplot(x="x", y="y", sort=False, kind="line", data=df)
sns.relplot(x="x", y="y", sort=True, kind="line", data=df)

fmri = sns.load_dataset("fmri") # multiple measurements for same x value
sns.relplot(x="timepoint", y="signal", kind="line", data=fmri) # with confidence interval
sns.relplot(x="timepoint", y="signal", ci=None, kind="line", data=fmri) # without confidence interval
sns.relplot(x="timepoint", y="signal", estimator=None, kind="line", data=fmri) # turn off aggregation