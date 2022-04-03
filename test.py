import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt


tmp = pd.DataFrame(0, index=[i for i in range(1, 10)], columns=[i for i in range(1, 4)])
tmp.iloc[5-1, 2-1] = 2  # cell [5][2]
tmp.iloc[4-1, 2-1] = 1  # cell [4][2]
tmp.iloc[2-1, 1-1] = 1  # cell [2][1]
tmp.iloc[3-1, 1-1] = 3  # cell [3][1]

print(tmp)
# counts, bins  = np.histogram(tmp.sum())
# plt.hist(bins[:-1], bins, weights=counts)
# plt.show()

hist = np.histogram(tmp.sum()[::600], bins= range(tmp.sum()[::600].min(),(tmp.sum())[::600].max())) 
_ = plt.plot(hist[1][:-1], hist[0], lw=2)
plt.show()



hist = tmp.sum().hist(histtype='stepfilled')
plt.show()

# matplotlib.plt.show()
# exmple
# counts, bins = np.histogram(data)
# plt.hist(bins[:-1], bins, weights=counts)

# exmaple 2
# x = [value1, value2, value3,....]
# plt.hist(x, bins = number of bins)
# plt.show()


# max_val = 5
# normalized_df= max_val*tmp/tmp.sum()
# print(normalized_df)


tmp.to_csv('tmp777.csv')


# tmp = np.zeros((10, 3), dtype=np.int)
# tmp[5, 2] = 999  # cell [5][2]
# # np.savetxt("tmp1115.csv", tmp, delimiter = ",", fmt="%.0f", header="A, B",  comments="")
# df = pd.DataFrame(tmp)
# df.drop([0], inplace=True, axis=0)
# df.drop([0], inplace=True, axis=1)
# df.to_csv('tmp999.csv')
print("Done")

