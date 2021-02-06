import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error #均方误差
from sklearn.metrics import mean_absolute_error #平方绝对误差
from sklearn.metrics import r2_score#R square

df = pd.read_csv('new_info_clean.csv')
points = []
for index,row in df.iterrows():
    points.append([row['my_tolerance'], row['Extension rate 22 '],
                  row['Decomposition rate 22']])

def compute_cost(w,b,points):
    total_cost = 0
    M = len(points)
    for i in range(M):
        x = points[i][0]
        y = points[i][1]
        z = points[i][2]
        total_cost += (z - (b + w[0]*x + w[1]*y))**2
    return total_cost/2

alpha = 0.00001
initial_w = [0,0]
initial_b = 0
num_iter = 500



def grad_desc(points, initial_w, initial_b, alpha, num_iter):
    w = initial_w
    b = initial_b
    # 定义一个list保存所有的损失函数值，用来显示下降过程。
    cost_list = []
    for i in range(num_iter):
        cost_list.append(compute_cost(w, b, points))
        w[0],w[1], b = step_grad_desc(w, b, alpha, points)
    return [w, b, cost_list]


def step_grad_desc(current_w, current_b, alpha, points):
    sum_grad_w1 = 0
    sum_grad_w2 = 0
    sum_grad_b = 0
    M = len(points)
    # 对每个点代入公式求和
    for i in range(M):
        x = points[i][0]
        y = points[i][1]
        z = points[i][2]
        sum_grad_w1 += (current_w[1] * y + current_w[0] * x + current_b - z) * x
        sum_grad_w2 += (current_w[1] * y + current_w[0] * x + current_b - z) * y
        sum_grad_b += current_w[1] * y + current_w[0] * x + current_b - y
    # 用公式求当前梯度
    # grad_w1 = 2 / M * sum_grad_w1
    # grad_w2 = 2 / M * sum_grad_w2
    # grad_b = 2 / M * sum_grad_b

    # 梯度下降，更新当前的w和b
    updated_w1 = current_w[0] - alpha * sum_grad_w1
    updated_w2 = current_w[1] - alpha * sum_grad_w2
    updated_b = current_b - alpha * sum_grad_b
    return updated_w1, updated_w2, updated_b

w, b, cost_list = grad_desc(points,initial_w,initial_b,alpha,num_iter)
y_predict = []
y_test = []
for index,row in df.iterrows():
    y_test.append(row['Decomposition rate 22'])
    x = row['my_tolerance']
    y = row['Extension rate 22 ']
    z = w[0]*x + w[1] * y + b
    y_predict.append(z)
print("MSE:" + str(mean_squared_error(y_test,y_predict)))
print("RMSE:" + str(np.sqrt(mean_squared_error(y_test,y_predict))))
print("MAE:" + str(mean_absolute_error(y_test,y_predict)))
print("R2:" + str(r2_score(y_test,y_predict)))