import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

fungi_number = 2
simulation_time = 1000
initialized_capacity = 100.0
c = 1.0

valuable_time = 0

x = np.zeros((fungi_number, simulation_time))
K = np.zeros(simulation_time)
growth_rate0 = np.zeros(fungi_number)
growth_rate_s = np.zeros((fungi_number, simulation_time))
r0 = np.zeros(fungi_number)
r_s = np.zeros((fungi_number, simulation_time))
tot_decomposition = np.zeros(simulation_time)   #all fungi decomposition in unit time


r_test = [0.008, 0.012]
x_test = [3.1, 0.2]
growth_test = [0.004, 0.003]

for i in range(fungi_number):
    r0[i] = r_test[i]
    x[i][0] = x_test[i]
    growth_rate_s[i][0] = growth_test[i]


for i in range(simulation_time):
    K[i] = initialized_capacity
    tmp_decomposition = 0.0
    tmp_tot_fungi = 0.0
    for j in range(fungi_number):
        if i > 0:
            tmp_decomposition += x[j][i-1] * r0[j]
            tmp_tot_fungi += x[j][i-1]
        else:
            tmp_decomposition = 0
            tmp_tot_fungi += x[j][i]
    tot_decomposition[i] = tmp_decomposition
    #print(tmp_decomposition)
    tmp_tot_decomposition = 0.0
    for j in range(i):
        tmp_tot_decomposition += tot_decomposition[j]
    print(tmp_tot_decomposition)
    if tmp_tot_decomposition > 70.0 and valuable_time == 0:
        valuable_time = i
    for j in range(fungi_number):
        growth_rate_s[j][i] = growth_rate_s[j][0] * (1.0 - tmp_tot_fungi/(initialized_capacity - c * tmp_tot_decomposition))
    if i > 0:
        for j in range(fungi_number):
            x[j][i] = x[j][i-1] * (1 + growth_rate_s[j][i])

s_time = np.arange(valuable_time)
x1_plot = np.zeros(valuable_time)
x2_plot = np.zeros(valuable_time)
for i in range(valuable_time):
    x1_plot[i] = x[0][i]
    x2_plot[i] = x[1][i]

print(valuable_time)
print(x1_plot)

plt.plot(s_time, x1_plot, 'r')
plt.plot(s_time, x2_plot, 'b')
plt.legend()
plt.savefig('different species.png')
plt.show()
