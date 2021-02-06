import pandas as pd
import scipy as sp
import numpy as np
from sklearn.metrics import mean_squared_error #均方误差
from sklearn.metrics import mean_absolute_error #平方绝对误差
from sklearn.metrics import r2_score#R square

df = pd.read_csv('new_info_clean.csv')
y_predict = []
y_test = []
for index,row in df.iterrows():
    y_test.append(row['Decomposition rate 22'])
    x = row['my_tolerance']
    y = row['Extension rate 22 ']
    z = 11.049511875508855+11.049511875508855*x+2.8546508129034187*y
    y_predict.append(z)
print("MSE:" + str(mean_squared_error(y_test,y_predict)))
print("RMSE:" + str(np.sqrt(mean_squared_error(y_test,y_predict))))
print("MAE:" + str(mean_absolute_error(y_test,y_predict)))
print("R2:" + str(r2_score(y_test,y_predict)))