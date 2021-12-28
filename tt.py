from matplotlib import pyplot as plt
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

data = pd.read_csv('data_pat.csv', sep=',')
end = 9850  # начало исследуемого участка
start = 7416  # начало обучающей выборки
s = end - start + 1
n = 20  # ширина окна
data_close = data.iloc[:, 4]
ticer = np.array(data_close)[start:start + s]

b = []
b1 = []
Amax_ticer = {}
window = {}
for i in range(start, end):
    for j in range(n - 10):
        data_close_ob = np.array(data_close)[end + j:end + n]
        data_close_ob_k = np.array(data_close)[i:i + n - j]
        kor_k = np.corrcoef(data_close_ob, data_close_ob_k)[0, 1]
        if j != 0 and j != n - 3:
            Amax_ticer.update({i: kor_k})
            window.update({i: j})
            # Amax_ticer.append(kor_k)
b.append(Amax_ticer)
b1.append([np.max(list(Amax_ticer.values())), list(Amax_ticer.keys())[list(Amax_ticer.values()).index(np.max(list(
    Amax_ticer.values())))]])  # массив содержащий индексы и максимальные значения коэффициентов корреляций при движении окна
b1[0].append(window[b1[0][1]])
print(b1)

f, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot([i for i in range(20)], data_close[b1[0][1]: b1[0][1] + 20], )
ax1.set_title('Sharing Y axis')
ax2.plot([i for i in range(20)], data_close[9850:9850 + 20])
plt.show()
