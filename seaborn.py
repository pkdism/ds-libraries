import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = 'darkgrid')

tips = sns.load_dataset("tips")

# scatterplot
sns.relplot(x = "total_bill", y = "tip", data = tips)

# scatterplot with colored points
sns.relplot(x = "total_bill", y = "tip", hue = "smoker", data = tips)

# scatterplot with colored points and different marker style for each class
sns.relplot(x = "total_bill", y = "tip", hue = "smoker", style = "smoker", data = tips)