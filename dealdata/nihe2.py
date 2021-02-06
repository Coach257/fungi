from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pandas as pd
from sklearn.metrics import mean_squared_error #均方误差
from sklearn.metrics import mean_absolute_error #平方绝对误差
from sklearn.metrics import r2_score#R square


def fun(x):  # 处理符号问题
    round(x, 2)
    if x >= 0:
        return '+' + str(x)
    else:
        return str(x)


# 主函数
if __name__ == '__main__':
    df = pd.read_csv('new_info_clean.csv')
    X = []
    Y = []
    Z = []
    y_predict = []
    y_test = []
    # 读入n个点的坐标
    for index,row in df.iterrows():
        X.append(row['my_tolerance'])
        Y.append(row['Extension rate 22 '])
        Z.append(row['Decomposition rate 22'])
    n = len(Z)

    # print(X)
    # print(Y)
    # print(Z)

    # fig = plt.figure() #建立一个新的图像
    # ax = Axes3D(fig) #建立一个三维坐标系
    # ax.scatter(X,Y,Z,c='b')
    # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
    # plt.show()

    # 求方程系数
    sigma_x = 0
    for i in X: sigma_x += i
    sigma_y = 0
    for i in Y: sigma_y += i
    sigma_z = 0
    for i in Z: sigma_z += i
    sigma_x2 = 0
    for i in X: sigma_x2 += i * i
    sigma_y2 = 0
    for i in Y: sigma_y2 += i * i
    sigma_x3 = 0
    for i in X: sigma_x3 += i * i * i
    sigma_y3 = 0
    for i in Y: sigma_y3 += i * i * i
    sigma_x4 = 0
    for i in X: sigma_x4 += i * i * i * i
    sigma_y4 = 0
    for i in Y: sigma_y4 += i * i * i * i
    sigma_x_y = 0
    for i in range(n):
        sigma_x_y += X[i] * Y[i]
    # print(sigma_xy)
    sigma_x_y2 = 0
    for i in range(n): sigma_x_y2 += X[i] * Y[i] * Y[i]
    sigma_x_y3 = 0
    for i in range(n): sigma_x_y3 += X[i] * Y[i] * Y[i] * Y[i]
    sigma_x2_y = 0
    for i in range(n): sigma_x2_y += X[i] * X[i] * Y[i]
    sigma_x2_y2 = 0
    for i in range(n): sigma_x2_y2 += X[i] * X[i] * Y[i] * Y[i]
    sigma_x3_y = 0
    for i in range(n): sigma_x3_y += X[i] * X[i] * X[i] * Y[i]
    sigma_z_x2 = 0
    for i in range(n): sigma_z_x2 += Z[i] * X[i] * X[i]
    sigma_z_y2 = 0
    for i in range(n): sigma_z_y2 += Z[i] * Y[i] * Y[i]
    sigma_z_x_y = 0
    for i in range(n): sigma_z_x_y += Z[i] * X[i] * Y[i]
    sigma_z_x = 0
    for i in range(n): sigma_z_x += Z[i] * X[i]
    sigma_z_y = 0
    for i in range(n): sigma_z_y += Z[i] * Y[i]
    # print("-----------------------")
    # 给出对应方程的矩阵形式
    a = np.array([[sigma_x4, sigma_x3_y, sigma_x2_y2, sigma_x3, sigma_x2_y, sigma_x2],
                  [sigma_x3_y, sigma_x2_y2, sigma_x_y3, sigma_x2_y, sigma_x_y2, sigma_x_y],
                  [sigma_x2_y2, sigma_x_y3, sigma_y4, sigma_x_y2, sigma_y3, sigma_y2],
                  [sigma_x3, sigma_x2_y, sigma_x_y2, sigma_x2, sigma_x_y, sigma_x],
                  [sigma_x2_y, sigma_x_y2, sigma_y3, sigma_x_y, sigma_y2, sigma_y],
                  [sigma_x2, sigma_x_y, sigma_y2, sigma_x, sigma_y, n]])
    b = np.array([sigma_z_x2, sigma_z_x_y, sigma_z_y2, sigma_z_x, sigma_z_y, sigma_z])
    # 高斯消元解线性方程
    res = np.linalg.solve(a, b)
    # print(a)
    # print(b)
    # print(x)
    # print("-----------------------")

    # 输出方程形式
    print("z=%.6s*x^2%.6s*xy%.6s*y^2%.6s*x%.6s*y%.6s" % (
    fun(res[0]), fun(res[1]), fun(res[2]), fun(res[3]), fun(res[4]), fun(res[5])))
    # 画曲面图和离散点
    fig = plt.figure()  # 建立一个空间
    ax = fig.add_subplot(111, projection='3d')  # 3D坐标

    n = 256
    u = np.linspace(-2,2)  # 创建一个等差数列
    v = np.linspace(0,10)
    x, y = np.meshgrid(u, v)  # 转化成矩阵

    # 给出方程
    z = res[0] * x * x + res[1] * x * y + res[2] * y * y + res[3] * x + res[4] * y + res[5]
    for index, row in df.iterrows():
        y_test.append(row['Decomposition rate 22'])
        x1 = row['my_tolerance']
        y1 = row['Extension rate 22 ']
        z1 = res[0] * x1 * x1 + res[1] * x1 * y1 + res[2] * y1 * y1 + res[3] * x1 + res[4] * y1 + res[5]
        y_predict.append(z1)
    print("MSE:" + str(mean_squared_error(y_test, y_predict)))
    print("RMSE:" + str(np.sqrt(mean_squared_error(y_test, y_predict))))
    print("MAE:" + str(mean_absolute_error(y_test, y_predict)))
    print("R2:" + str(r2_score(y_test, y_predict)))
    # 画出曲面
    ax.plot_surface(x, y, z, rstride=3, cstride=3, cmap=cm.jet)
    # 画出点
    ax.scatter(X, Y, Z, c='r')
    plt.show()

