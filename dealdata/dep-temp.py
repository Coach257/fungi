from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error #均方误差
from sklearn.metrics import mean_absolute_error #平方绝对误差
from sklearn.metrics import r2_score#R square

df = pd.read_csv('new_info_clean.csv')
x = []
y = []
d10 = []
d16 = []
d22 = []

def f_fit(x,y_fit):
    b,c=y_fit.tolist()
    y = []
    for i in x:
        y.append(b * i + c)
    return y

for index,row in df.iterrows():
    d10.append(row['Decomposition rate 10 '])
    d16.append(row['Decomposition rate 16 '])
    d22.append(row['Decomposition rate 22'])

for index,row in df.iterrows():
    x.extend([10,16,22])
    gavg = (row['Decomposition rate 10 '] * row['Decomposition rate 16 '] * row['Decomposition rate 22'])**(1.0/3.0)
    y.extend([row['Decomposition rate 10 ']/gavg,
              row['Decomposition rate 16 ']/gavg,
              row['Decomposition rate 22']/gavg])
y_fit = np.polyfit(x,y,1)
y_show = np.poly1d(y_fit)
print(y_show)
y1=f_fit(x,y_fit)
y_predict = y1
y_test = y
print("MSE:" + str(mean_squared_error(y_test, y_predict)))
print("RMSE:" + str(np.sqrt(mean_squared_error(y_test, y_predict))))
print("MAE:" + str(mean_absolute_error(y_test, y_predict)))
print("R2:" + str(r2_score(y_test, y_predict)))

plt.scatter(x,y,c='b')#散点图
plt.plot(x,y1,'r--')
plt.xlabel('Temporatrue')
plt.ylabel('Decomposition rate/geo mean')
plt.savefig('logdep-temp.png')
plt.show()

