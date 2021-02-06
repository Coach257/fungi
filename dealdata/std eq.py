import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error #均方误差
from sklearn.metrics import mean_absolute_error #平方绝对误差
from sklearn.metrics import r2_score#R square

df = pd.read_csv('new_info_clean.csv')
x = []
y = []
for index,row in df.iterrows():
    x.append([1,row['my_tolerance'], row['Extension rate 22 ']])
    y.append(row['Decomposition rate 22'])
x = np.mat(x)
y = np.mat(y)
y = y.T

w = ((x.T*x).I)*(x.T)*y
w = w.tolist()
y_predict = []
y_test = []
for index,row in df.iterrows():
    y_test.append(row['Decomposition rate 22'])
    x = row['my_tolerance']
    y = row['Extension rate 22 ']
    z = w[0][0] + w[1][0] * x + w[2][0] * y
    y_predict.append(z)
print("MSE:" + str(mean_squared_error(y_test,y_predict)))
print("RMSE:" + str(np.sqrt(mean_squared_error(y_test,y_predict))))
print("MAE:" + str(mean_absolute_error(y_test,y_predict)))
print("R2:" + str(r2_score(y_test,y_predict)))
