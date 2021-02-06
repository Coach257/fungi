from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

#定义3D画面
fig = plt.figure()
ax = Axes3D(fig)

#读取表格数据
point = []
df = pd.read_csv('new_info_clean.csv')
for index,row in df.iterrows():
    point.append([row['my_tolerance'],row['Extension rate 22 '],
                  row['Decomposition rate 22']])
plt.xlabel("my_tolerance")
plt.ylabel("Extension rate")

#表示矩阵中的值
Isum = 0.0
x1sum = 0.0
x2sum = 0.0
x1_2sum = 0.0
x1x2sum = 0.0
x2_2sum = 0.0
ysum = 0
x1ysum = 0
x2ysum = 0

#在图中显示各店的位置
for i in range(0,len(point)):
    x1i = point[i][0]
    x2i = point[i][1]
    yi = point[i][2]
    ax.scatter(x1i,x2i,yi,color = "red")
    # show_point = "[" + str(x1i) + "," + str(x2i) + "," + str(yi) + "]"
    # ax.text(x1i,x2i,yi,show_point)
    Isum = Isum + 1
    x1sum = x1sum + x1i
    x2sum = x2sum + x2i
    x1_2sum = x1_2sum + x1i**2
    x1x2sum = x1x2sum + x1i*x2i
    x2_2sum = x2_2sum + x2i**2
    ysum = ysum + yi
    x1ysum = x1ysum + x1i*yi
    x2ysum = x2ysum + x2i*yi

#进行矩阵运算
m1 = [[Isum,x1sum,x2sum],[x1sum,x1_2sum,x1x2sum],[x2sum,x1x2sum,x2_2sum]]
mat1 = np.matrix(m1)
m2 = [[ysum],[x1ysum],[x2ysum]]
mat2 = np.matrix(m2)
_mat1 = mat1.getI()
mat3 = _mat1 * mat2
m3 = mat3.tolist()
a0 = m3[0][0]
a1 = m3[1][0]
a2 = m3[2][0]

#绘制回归线
x1 = np.linspace(-2,2)
x2 = np.linspace(0,10)
y = a0 + a1*x1 + a2*x2
print("y=" + str(a0) + "+" + str(a0) + "*x1" + "+" + str(a2) + "*x2")
ax.plot(x1,x2,y)
show_line = "Decomposition rate=" + str(a0) + "+" + str(a1) + "my_tolerance" + "+" + str(a2) + "Extension rate"
plt.title(show_line)
plt.savefig('line.png')
plt.show()



