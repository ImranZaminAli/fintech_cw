import os
import pandas as pd
from statistics import mean
import numpy as np
from cwhelpers import run_stats
import seaborn as sns
import matplotlib.pyplot as plt
n = 60
initial_seed = 100
# pps = pd.DataFrame()
# stat_summary = pd.DataFrame()
# filename = f'{initial_seed + 0}_strats.csv'
# file = open(filename, 'r')
# lines = file.readlines()
# file.close()
# print(np.average([float(value.split(',')[15].strip()) for value in lines[-5:]]))
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
# print(pps.head())
# fig, axes = plt.subplots(1,2)
# sns.kdeplot(data=pps, fill=False, ax=axes[0])
# sns.boxplot(data=pps, ax=axes[1])
# plt.show()
# summary_entry = run_stats(pps, {})
# summary_entry = pd.DataFrame([summary_entry])
# stat_summary = pd.concat([stat_summary, summary_entry], ignore_index=True)
# for i in range(20):
#     pps = pd.DataFrame()
#     fig, ax = plt.subplots(1,2, sharey=True)
#     #filename = os.path.join('d1', f'{initial_seed + 1}_strats.csv')
#     filename = f'{initial_seed + i}_strats.csv'
#     file = open(filename, 'r')
#     lines = file.readlines()
#     file.close()
#     y = 'mbuy'
#     for j in range(len(lines)):
#         line = lines[j].split(',')
#         entry = {'t / simulated seconds' : [float(line[1].strip())], y : [float(line[15].strip())]}
#         entry = pd.DataFrame(entry)
#         pps = pd.concat([pps, entry], ignore_index=True)
#     pps.plot.scatter(x='t / simulated seconds', y=y, ax = ax[0])
#     pps1 = pd.DataFrame()
#     filename = os.path.join('tabu', f'{initial_seed + 1}_strats.csv')
#     file = open(filename, 'r')
#     lines = file.readlines()
#     file.close()
#     for j in range(len(lines)):
#         line = lines[j].split(',')
#         entry = {'t / simulated seconds' : [float(line[1].strip())], y : [float(line[15].strip())]}
#         entry = pd.DataFrame(entry)
#         pps1 = pd.concat([pps1, entry], ignore_index=True)
#     pps1.plot.scatter(x='t / simulated seconds', y=y, ax = ax[1])
#     #print(i)
#     plt.show()

# initial seed of 13
# pps = pd.DataFrame()
# for i in range(n):
#     entry = {}
#     filename = f'{initial_seed + i}_strats.csv'

#     sh = open(os.path.join('d1', filename), 'r')
#     lines = sh.readlines()
#     sh.close()
#     entry['sh'] = [float(lines[-1].split(',')[15].strip())]

#     tabu = open(filename, 'r')
#     lines = tabu.readlines()
#     tabu.close()
#     entry['tabu'] = [float(lines[-1].split(',')[15].strip())]
#     #print('tabu')
#     entry = pd.DataFrame(entry)
#     pps = pd.concat([pps, entry], ignore_index=True)


# #print(run_stats(pps, {}))

# sns.kdeplot(data=pps, fill=False)
# plt.show()


# pps = pd.DataFrame()
# folders = ['d1_100', 'tabu']
# for i in range(20):
#     entry = {}
#     filename = f'{initial_seed + i}_strats.csv'
    
#     for folder in folders:
#         file = open(os.path.join(folder, filename))
#         lines = file.readlines()
#         file.close()
#         entry[folder] = [np.average([float(value.split(',')[15].strip()) for value in lines[-5:]])]
#         entry = pd.DataFrame(entry)

#     pps = pd.concat([pps, entry], ignore_index=True)

# sns.kdeplot(data=pps, fill=False)
# plt.show()
# print(run_stats(pps, {}))

pps = pd.DataFrame()
for i in range(n):
    entry = {}
    filename = f'{initial_seed + i}_strats.csv'
    file = open(filename, 'r')
    lines = file.readlines()
    print(len(lines[0]))
    file.close()
    init_lines = lines[5:]
    ave = []
    for init in init_lines:
        pps_str = init.split(',')[15]
        pps_val = float(pps.str.strip())
        ave.append(pps_val)


    #entry['initial'] = [np.average([float(x.split(',')[15].strip()) for x in lines[5:]])]