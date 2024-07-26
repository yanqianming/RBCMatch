import numpy as np
import matplotlib.pyplot as plt


font = {'family': 'Arial', 'size': 16}
plt.rc('font', **font)

n_groups = 4
means_1 = (0.133, 0.044, 0.557, 0.266)
std_1 = (0.026, 0.013, 0.099, 0.081)

means_2 = (0.384, 0.040, 0.053, 0.523)
std_2 = (0.028, 0.014, 0.025, 0.013)

means_3 = (0.631, 0.043, 0.115, 0.211)
std_3 = (0.057, 0.011, 0.023, 0.056)

means_4 = (0.932, 0.029, 0.021, 0.018)
std_4 = (0.049, 0.029, 0.014, 0.012)


color = ["#{:02x}{:02x}{:02x}".format(55, 103, 149),
         "#{:02x}{:02x}{:02x}".format(114, 188, 213),
         "#{:02x}{:02x}{:02x}".format(255, 208, 98),
         "#{:02x}{:02x}{:02x}".format(231, 98, 84)]


fig, ax = plt.subplots(figsize=(8, 5), dpi=600)
index = np.arange(n_groups)
bar_width = 0.2

opacity = 0.8

rects1 = ax.bar(index, means_1, bar_width, capsize=5,
                alpha=opacity, color=color[0], ecolor='k',
                yerr=std_1, edgecolor='k',
                label='Day 4')

rects2 = ax.bar(index + bar_width, means_2, bar_width, capsize=5,
                alpha=opacity, color=color[1], ecolor='k',
                yerr=std_2, edgecolor='k',
                label='Day 6')

rects3 = ax.bar(index + 2 * bar_width, means_3, bar_width, capsize=5,
                alpha=opacity, color=color[2], ecolor='k',
                yerr=std_3, edgecolor='k',
                label='Day 8')

rects4 = ax.bar(index + 3 * bar_width, means_4, bar_width, capsize=5,
                alpha=opacity, color=color[3], ecolor='k',
                yerr=std_4, edgecolor='k',
                label='Control')


ax.set_ylabel('Cell proportion', font)
ax.set_xticks(index + 0.3)
ax.set_xticklabels(('Normal\nRBCs', 'Micronucleus\nresidue', 'Crenated\nRBCs', 'Purple\nRBCs'))
ax.legend(loc='best', ncol=2, frameon=False)
plt.show()
