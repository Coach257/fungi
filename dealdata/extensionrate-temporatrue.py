from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error #均方误差
from sklearn.metrics import mean_absolute_error #平方绝对误差
from sklearn.metrics import r2_score#R square

rate_max = 0
def func(x,p,a,w):
    return (p*x**2+a*x+w)*rate_max

def geomean(x):
    product = 0.0
    for i in x:
        product += i
    product /= len(x)
    return product

df = pd.read_csv('Fungi_temperature_curves.csv')
rate_fit = []
p1 = []
p2 = []
n = df['species'].nunique()
pp1 = []
aa1 = []
ww1 = []
limit_num = 0.1
for name,group in df.groupby('species'):
    rate_max = 0
    T_best = 0.0
    x1 = []
    y1 = []
    for index,row in group.iterrows():
        if (row['hyphal_rate'] > rate_max) :
            rate_max = row['hyphal_rate']
            T_best = row['temp_c']
    for index,row in group.iterrows():
        if (row['temp_c'] <= T_best):
            x1.append(row['temp_c'])
            y1.append(row['hyphal_rate'] / rate_max)
    x1 = np.array(x1)
    y1 = np.array(y1)
    # popt,pcov = curve_fit(func,x1,y1,bounds=([-1,-1,-1,-1,-1,-1],[1,1,1,1,1,1]))
    # p_predict.append(popt[0])
    # q_predict.append(popt[1])
    # a_predict.append(popt[2])
    # b_predict.append(popt[3])
    # w_predict.append(popt[4])
    # c_predict.append(popt[5])
    # rate_fit.append(rate_max)
    f1 = np.polyfit(x1,y1,2)
    pp1.append(f1[0])
    aa1.append(f1[1])
    ww1.append(f1[2])
    p1.append(np.poly1d(f1))

pp1_guess = geomean(pp1)
aa1_guess = geomean(aa1)
ww1_guess = geomean(ww1)
# p1_guess = geomean(p_predict)
# q1_guess = geomean(q_predict)
# a1_guess = geomean(a_predict)
# b1_guess = geomean(b_predict)
# w1_guess = geomean(w_predict)
# c1_guess = geomean(c_predict)
print(pp1_guess,aa1_guess,ww1_guess)
pp1 = []
aa1 = []
ww1 = []

for name,group in df.groupby('species'):
    rate_max = 0
    T_best = 0.0
    x1 = []
    y1 = []
    for index,row in group.iterrows():
        if (row['hyphal_rate'] > rate_max) :
            rate_max = row['hyphal_rate']
            T_best = row['temp_c']
    for index,row in group.iterrows():
        if (row['temp_c'] >= T_best):
            x1.append(row['temp_c'])
            y1.append(row['hyphal_rate'] / rate_max)
    x1 = np.array(x1)
    y1 = np.array(y1)
    # popt,pcov = curve_fit(func,x1,y1,bounds=([-1,-1,-1,-1,-1,-1],[1,1,1,1,1,1]))
    # p_predict.append(popt[0])
    # q_predict.append(popt[1])
    # a_predict.append(popt[2])
    # b_predict.append(popt[3])
    # w_predict.append(popt[4])
    # c_predict.append(popt[5])
    f1 = np.polyfit(x1, y1, 2)
    pp1.append(f1[0])
    aa1.append(f1[1])
    ww1.append(f1[2])
    p2.append(np.poly1d(f1))

# p2_guess = geomean(p_predict)
# q2_guess = geomean(q_predict)
# a2_guess = geomean(a_predict)
# b2_guess = geomean(b_predict)
# w2_guess = geomean(w_predict)
# c2_guess = geomean(c_predict)
pp2_guess = geomean(pp1)
aa2_guess = geomean(aa1)
ww2_guess = geomean(ww1)
print(pp2_guess,aa2_guess,ww2_guess)
#
spcnt = -1
for name,group in df.groupby('species'):
    x = []
    y = []
    spcnt += 1
    y_predict = []
    rate_max = 0
    T_best = 0.0
    for index,row in group.iterrows():
        if (row['hyphal_rate'] > rate_max) :
            rate_max = row['hyphal_rate']
            T_best = row['temp_c']
    for index, row in group.iterrows():
        if (row['hyphal_rate'] >= 0.05):
            x.append(row['temp_c'])
            y.append(row['hyphal_rate'])
            if (row['temp_c'] <= T_best):
                y_predict.append(func(row['temp_c'],pp1_guess,aa1_guess,ww1_guess))
            else:
                y_predict.append(func(row['temp_c'],pp2_guess,aa2_guess,ww2_guess))
    plt.title(row['species'])
    plt.xlabel('temp_c')
    plt.ylabel('hyphal_rate')
    plt.scatter(x,y,c='b',s=0.1)
    plt.scatter(x,y_predict,c='r',s=0.1)
    plt.savefig(row['species'] + ".png")
    plt.show()
