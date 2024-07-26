import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# csv_path = "../../results/20240511-pb-fixmatch-66/rbc_all_ftrs.csv"
# df = pd.read_csv(csv_path)

# prop = 'circularity'
# plt.rc('font', family="Arial", size=16)
# plt.gcf().subplots_adjust(bottom=0.14, left=0.1)
# legends = ['Contrast', '4 days', '6 days', '8 days']
# for day in range(4):
#     selected = df[df['day'] == day][prop]
#     sns.kdeplot(selected, fill=True, label=legends[day])
#
# plt.legend(loc="upper left")
# plt.xlabel('Roundness')
# plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
# plt.show()


np_path = "../../results/20240511-pb-fixmatch-66/representations2d.npy"
representations = np.load(np_path)
plt.rc('font', family="Arial", size=18)
plt.gcf().subplots_adjust(bottom=0.14, left=0.14)
legends = ['Control', 'Day 4', 'Day 6', 'Day 8']
for day in range(4):
    selected = representations[representations[:, 3] == day, 0]
    sns.kdeplot(selected, fill=True, label=legends[day])

plt.legend(loc="upper right")
plt.xlabel('Principal Component 1')
plt.ylabel('Probability')
plt.yticks([0, 0.02, 0.04, 0.06])
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
plt.show()
