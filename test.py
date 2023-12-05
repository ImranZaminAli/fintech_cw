import os
import pandas as pd
from statistics import mean
import numpy as np
from cwhelpers import run_stats
import seaborn as sns
import matplotlib.pyplot as plt
n = 60
initial_seed = 100
stat_summary = pd.DataFrame()
# for i in range(n):
#     start_entry = []
#     end_entry = []
    
#     filename = f'{initial_seed + i}_strats.csv'
#     file = open(filename, 'r')
#     lines = file.readlines()
#     file.close()
#     start_entry = np.average([float(value.split(',')[15].strip()) for value in lines[:5]])
#     end_entry = np.average([float(value.split(',')[15].strip()) for value in lines[-5:]])
#     pps_entry = pd.DataFrame({'start': [start_entry], 'end': [end_entry]})
    
#     pps = pd.concat([pps, pps_entry], ignore_index=True)
# summary_entry = run_stats(pps, {})
# summary_entry = pd.DataFrame([summary_entry])
# stat_summary = pd.concat([stat_summary, summary_entry], ignore_index=True)
# fig, axes = plt.subplots(1,2)
# sns.kdeplot(data=pps, fill=False, ax=axes[0])
# sns.boxplot(data=pps, ax=axes[1])
# plt.show()

for i in range(n):
    pps = pd.DataFrame()
    filename = os.path.join('d1', f'{initial_seed + i}_strats.csv')
    file = open(filename, 'r')
    lines = file.readlines()
    file.close()
    y = 'mbuy'
    for j in range(len(lines)):
        line = lines[j].split(',')
        entry = {'t / simulated seconds' : [float(line[1].strip())], y : [float(line[15].strip())]}
        entry = pd.DataFrame(entry)
        pps = pd.concat([pps, entry], ignore_index=True)

    pps.plot.scatter(x='t / simulated seconds', y=y)
    #print(i)
    plt.show()

# initial seed of 13
