import os
import pandas as pd
from statistics import mean
import numpy as np
from cwhelpers import run_stat_tests
n = 30
initial_seed = 100
pps = pd.DataFrame()
for i in range(n):
    start_entry = []
    end_entry = []
    
    filename = f'{initial_seed + i}_strats.csv'
    file = open(filename, 'r')
    lines = file.readlines()
    file.close()
    start_entry = np.average([float(value.split(',')[19].strip()) for value in lines[:5]])
    end_entry = np.average([float(value.split(',')[19].strip()) for value in lines[-5:]])
    pps_entry = pd.DataFrame({'start': [start_entry], 'end': [end_entry]})
    
    pps = pd.concat([pps, pps_entry], ignore_index=True)

run_stat_tests(pps)