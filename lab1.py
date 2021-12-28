import matplotlib.pyplot as plt
import numpy as np

from nd import nep_aprox, err_ex

sample_len = 200

data_x = np.arange(0, 5, 5 / sample_len)
noise = np.random.normal(0, 0.3, sample_len)
data_y = [0.1 * (data_x[i]) * np.cos(data_x[i]) + 0.5 * data_x[i] + noise[i] for i in range(len(data_x))]
real_data = [0.1 * (data_x[i]) * np.cos(data_x[i]) + 0.5 * data_x[i] for i in range(len(data_x))]
temp = np.random.uniform(0, 1, sample_len)
x3 = [i < 0.4 for i in temp]

# Вход в модель

y1 = [data_y[i] for i in range(len(data_y)) if x3[i]]
y2 = [data_y[i] for i in range(len(data_y)) if not x3[i]]

data_1 = [data_x[i] for i in range(len(data_x)) if x3[i]]
data_2 = [data_x[i] for i in range(len(data_x)) if not x3[i]]

min_err = False
min_c = 0
for i in np.arange(1, 5.2, 0.2):
    temp_err_1 = err_ex(nep_aprox([[data_x[i] for i in range(len(data_x)) if x3[i]], y1], i, range(len(y1))),
                        [real_data[i] for i in range(len(real_data)) if x3[i]])
    temp_err_2 = err_ex(nep_aprox([[data_x[i] for i in range(len(data_x)) if not x3[i]], y2], i, range(len(y2))),
                        [real_data[i] for i in range(len(real_data)) if not x3[i]])
    if not min_err or temp_err_1 + temp_err_2 < min_err:
        min_err = temp_err_1 + temp_err_2
        min_c = i

res1 = nep_aprox([[data_x[i] for i in range(len(data_x)) if x3[i]], y1], min_c, range(len(y1)))

res2 = nep_aprox([[data_x[i] for i in range(len(data_x)) if not x3[i]], y2], min_c, range(len(y2)))

plt.scatter([data_x[i] for i in range(len(data_x)) if x3[i]], y1)
plt.plot([data_x[i] for i in range(len(data_x)) if x3[i]], res1)
plt.show()
plt.cla()




plt.scatter([data_x[i] for i in range(len(data_x)) if not x3[i]], y2)
plt.plot([data_x[i] for i in range(len(data_x)) if not x3[i]], res2)
plt.show()
print(min_c)
