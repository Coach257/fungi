from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error #均方误差
from sklearn.metrics import mean_absolute_error #平方绝对误差
from sklearn.metrics import r2_score#R square

df = pd.read_csv('Fungi_moisture_curves.csv')
x = []
y = []
print(df['species'].nunique())
for index,row in df.iterrows():
    x.append(row['matric_pot'])
    y.append(row['hyphal_rate'])
plt.scatter(x,y,c='b',s=0.1)#散点图
plt.show()