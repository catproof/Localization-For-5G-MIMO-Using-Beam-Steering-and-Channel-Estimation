import os
import string
from threading import Thread
import numpy as np
import numpy.random
import math
import random
import pandas as pd
from matplotlib import pyplot as plt

from util import norm_2, print_debug

exp_folder_lists = ["first experiments", "second experiments", "third experiments"]
df_list = []
for exp_folder in exp_folder_lists:
  csv_file = os.path.join(os.getcwd(), exp_folder, "data.csv")
  df_list.append(pd.read_csv(csv_file))

all_data_arrays = pd.concat(df_list).to_numpy(dtype=np.double)
print("we got all datas \n{}".format(all_data_arrays))

x = all_data_arrays[:, 0] / all_data_arrays[:, 1]
posRx_error_means = all_data_arrays[:, 2]
posRs_error_intervals = all_data_arrays[:, 3]
alpha_error_means = all_data_arrays[:, 4]
alpha_error_intervals = all_data_arrays[:, 5]

for i in range(0, np.size(x)):
  plt.plot(x[i], posRx_error_means[i], color='b', marker='o')
  plt.plot(x[i], posRx_error_means[i] + posRs_error_intervals[i], color='b', marker='v', alpha=.1)
  plt.plot(x[i], posRx_error_means[i] - posRs_error_intervals[i], color='b', marker='^', alpha=.1)

plt.xlabel("d / lambda_n")
plt.ylabel("Localization Error")
plt.savefig(os.path.join(os.getcwd(), "d div lambda_n scatter plot for posRx.jpg"),pil_kwargs={'quality':95}, dpi=300)
plt.show()
plt.close()


for i in range(0, np.size(x)):
  plt.plot(x[i], alpha_error_means[i], color='r', marker='o')
  plt.plot(x[i], alpha_error_means[i] + alpha_error_intervals[i], color='r', marker='v', alpha=.1)
  plt.plot(x[i], alpha_error_means[i] - alpha_error_intervals[i], color='r', marker='^', alpha=.1)

plt.xlabel("d / lambda_n")
plt.ylabel("Orientation Error (in radian)")
plt.savefig(os.path.join(os.getcwd(), "d div lambda_n scatter plot for alpha.jpg"),pil_kwargs={'quality':95}, dpi=300)
plt.show()
plt.close()



