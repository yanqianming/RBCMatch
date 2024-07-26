import numpy as np
import matplotlib.pyplot as plt
from matplotlib import container
from matplotlib.ticker import NullLocator

# color1 = "#038355"  # 孔雀绿
# color2 = "#ffc34e"  # 向日黄
color1 = "#ae5e52"
color2 = "#6b9ac8"
x = [1, 2, 5, 10, 25, 50, 75, 100]
fix_precision_mean = [0.804, 0.806, 0.816, 0.819, 0.829, 0.832, 0.830, 0.821]
fix_precision_std = [0.01, 0.003, 0.004, 0.001, 0.004, 0.002, 0.005, 0]
doan_precision_mean = [0.731, 0.776, 0.787, 0.804, 0.806, 0.819, 0.823, 0.821]
doan_precision_std = [0.01, 0.024, 0.012, 0.008, 0.004, 0.004, 0.001, 0]

plt.figure(figsize=(8, 6))
# plt.gcf().subplots_adjust(bottom=0.18, left=0.15)
plt.rc('font', family="Arial", size=16)
plt.errorbar(x, fix_precision_mean, yerr=fix_precision_std, fmt="o-", ecolor=color1, color=color1, elinewidth=2, capsize=4, label="RBCMatch")
plt.errorbar(x, doan_precision_mean, yerr=doan_precision_std, fmt="s-", ecolor=color2, color=color2, elinewidth=2, capsize=4, label="Doan et al.")
ax = plt.gca()
handles, labels = ax.get_legend_handles_labels()
handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]
ax.legend(handles, labels, loc="lower right")
plt.xscale("log")
plt.ylim([0.7, 0.84])
plt.yticks(np.arange(0.7, 0.84, 0.04))
plt.gca().set_xticks(x)
plt.gca().set_xticklabels(x)
plt.xlabel("Percentage of Labeled Samples(%)")
plt.ylabel("Weighted Precision")
plt.show()


fix_recall_mean = [0.734, 0.745, 0.747, 0.759, 0.767, 0.761, 0.761, 0.768]
fix_recall_std = [0.017, 0.020, 0.006, 0.018, 0.009, 0.015, 0.003, 0]
doan_recall_mean = [0.647, 0.693, 0.746, 0.742, 0.752, 0.773, 0.769, 0.768]
doan_recall_std = [0.029, 0.043, 0.022, 0.011, 0.014, 0.005, 0.009, 0]

plt.figure(figsize=(8, 6))
# plt.gcf().subplots_adjust(bottom=0.18, left=0.15)
plt.rc('font', family="Arial", size=16)
plt.errorbar(x, fix_recall_mean, yerr=fix_recall_std, fmt="o-", ecolor=color1, color=color1, elinewidth=2, capsize=4, label="RBCMatch")
plt.errorbar(x, doan_recall_mean, yerr=doan_recall_std, fmt="s-", ecolor=color2, color=color2, elinewidth=2, capsize=4, label="Doan et al.")
ax = plt.gca()
handles, labels = ax.get_legend_handles_labels()
handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]
ax.legend(handles, labels, loc="lower right")
plt.xscale("log")
plt.ylim([0.6, 0.8])
plt.yticks(np.arange(0.6, 0.78, 0.04))
plt.gca().set_xticks(x)
plt.gca().set_xticklabels(x)
plt.xlabel("Percentage of Labeled Samples(%)")
plt.ylabel("Weighted Recall (equals to Acc.)")
plt.show()


fix_f1_mean = [0.751, 0.761, 0.764, 0.776, 0.782, 0.778, 0.776, 0.782]
fix_f1_std = [0.013, 0.016, 0.005, 0.015, 0.007, 0.012, 0.002, 0]
doan_f1_mean = [0.661, 0.708, 0.756, 0.759, 0.766, 0.786, 0.781, 0.782]
doan_f1_std = [0.025, 0.037, 0.021, 0.01, 0.009, 0.003, 0.008, 0]

plt.figure(figsize=(8, 6))
# plt.gcf().subplots_adjust(bottom=0.18, left=0.15)
plt.rc('font', family="Arial", size=16)
plt.errorbar(x, fix_f1_mean, yerr=fix_f1_std, fmt="o-", ecolor=color1, color=color1, elinewidth=2, capsize=4, label="RBCMatch")
plt.errorbar(x, doan_f1_mean, yerr=doan_f1_std, fmt="s-", ecolor=color2, color=color2, elinewidth=2, capsize=4, label="Doan et al.")
ax = plt.gca()
handles, labels = ax.get_legend_handles_labels()
handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]
ax.legend(handles, labels, loc="lower right")
plt.xscale("log")
plt.ylim([0.6, 0.8])
plt.yticks(np.arange(0.6, 0.78, 0.04))
plt.gca().set_xticks(x)
plt.gca().set_xticklabels(x)
plt.xlabel("Percentage of Labeled Samples(%)")
plt.ylabel("Weighted F1 Score")
plt.show()
