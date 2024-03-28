import seaborn as sns
import matplotlib.pyplot as plt

import pandas
grid = sns.JointGrid(pandas.DataFrame(data={'mz':[1,2,3,4,5,6,7,8,9], 'rt':[20,30,40,50,60,70,80,90,100]}),\
	space=1, height=6, ratio=50,\
	xlim=(1, 9),\
	ylim=(20, 100))
grid.plot_joint(plt.scatter, color='blue', alpha=.8)