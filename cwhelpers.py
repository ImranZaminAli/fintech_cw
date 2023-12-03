import pandas as pd
import os
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import random
import pingouin as pg
import statsmodels
from BSE import market_session
from os import cpu_count
import numpy as np
from multiprocessing import Pool

def anova_test(profits):
	anova = pg.rm_anova(data=profits, correction=True)
	print('Testing for sphericity with Mauchlys Test')
	sphericity = anova['p-spher'][0]
	statistic = anova['F'][0]
	pvalue = None
	if anova['sphericity'][0]:
		pvalue = anova['p-unc'][0]
		print(f'Do not reject null hyptothesis (p={sphericity}). There is enough evidence to suggest the samples have sphericity')
	else:
		pvalue = anova['p-GG-corr'][0]
		print(f'Reject null hyptothesis (p={sphericity}). There is insufficient evidence to suggest the samples have sphericity. Applying Greenhouse-Gessier correction.')
	# if anova['sphericity']:
	#     print(f'Do not reject null hypothesis {p={}}')
	return statistic, pvalue

def run_stat_tests(profits):
	for col in profits.columns:
		print(f'{col}: mean={profits[col].mean()} std={profits[col].std()}')
		print('Check each column if the data is normally distrubuted with the shapiro wilk test:')
	is_normal = True
	# check if the data is normally distributed
	for col in profits.columns:
		_, pvalue = stats.shapiro(profits[col])
		if pvalue < 0.05:
			print(f'Condition: {col}. Reject null hypothesis (p={pvalue}). There is enough evidence to suggest the data is not normally distributed')
			is_normal = False
		else:
			print(f'Condition: {col}. Do not reject null hypothesis (p={pvalue}). There is enough evidence to suggest the data is normally distributed')

	columns = [profits[column] for column in profits.columns]
	is_two_dist = len(columns) == 2
	if is_two_dist:
		fig, axes = plt.subplots(1,2)
		sns.kdeplot(data=profits, fill=False, ax=axes[0])
		sns.boxplot(data=profits, ax=axes[1])
		plt.show()
		sns.violinplot(data=profits, inner='box')
		plt.show()
	else:
		sns.violinplot(data=profits, inner='box')
		plt.show()
	pvalue = None
	further_test = False
	#if is_normal:
		#test_name = 'paired t-test' if is_two_dist else 'Repeated measures ANOVA test'
		#_, pvalue = stats.ttest_rel(*columns) if is_two_dist else anova_test(profits)
	#else:
		#test_name = 'Wilcoxon Signed Rank test' if is_two_dist else 'Friedman test'
		#_, pvalue = stats.wilcoxon(*columns) if is_two_dist else stats.friedmanchisquare(*columns)
	if is_two_dist:
		test_name = 'paired t-test' if is_normal else 'Wilcoxon Signed Rank test'
		_, pvalue = stats.ttest_rel(*columns) if is_normal else stats.wilcoxon(*columns)
	else:
		test_name = 'Repeated measures ANOVA test' if is_normal else 'Friedman test'
		_, pvalue = anova_test(profits)  if is_normal else stats.friedmanchisquare(*columns)
		further_test = pvalue < 0.05

	print(f'There are {len(columns)} distributions. Therefore using {test_name}')
	if pvalue < 0.05:
		print(f'Reject null hypothesis (p={pvalue}). There is sufficient evidence to suggest groups have a population different mean')
	else:
		print(f'Do not reject null hypothesis (p={pvalue}). There is not enough evidence to suggest the groups have a different population mean')

	if further_test:
		# converting to long format
		column_names = list(profits.columns)
		profits['Experiment'] = range(len(profits))
		long = pd.melt(profits, id_vars='Experiment', value_vars=column_names)
		pairwise_tests = long.pairwise_tests(dv='value', within='variable', subject='Experiment', padjust='holm', parametric=is_normal)
		indexes = [i for i in range(len(pairwise_tests)) if pairwise_tests['p-corr'][i] < 0.05]
		selected_rows = pairwise_tests.loc[indexes]
		result = list(zip(selected_rows['A'], selected_rows['B'], selected_rows['p-corr']))
		print(result)


# given a tuple of the algos, a tuple of the percentages for that trader and the number of traders return the specs
def get_traders_specs(algos, percentages, num_traders):
	if sum(percentages) != 100:
		raise Exception('The percentages should add to 100')
	if len(algos) != len(percentages):
		raise Exception(f'algos len: {len(algos)} %s len: {len(percentages)} are not equal')

	num_traders_each = tuple(round(percent * num_traders / 100) for percent in percentages)
	if sum(num_traders_each) != num_traders:
		raise Exception('num traders each != num traders')
	return list(zip(algos, num_traders_each))


def run_sessions(args):
	market_args, initial_seed, instance= args
	print(f'started {instance}', flush = True)
	seed = initial_seed + instance
	random.seed(seed)
	market_session(str(seed), *market_args)
	print(f'finished {instance}', flush = True)

	
def part_d1(n, market_args, initial_seed):
	
	pps = pd.DataFrame()
	with Pool(cpu_count()) as p:
		p.map(run_sessions, [(market_args, initial_seed, i) for i in range(n)])
		p.close()
		p.join()

	for i in range(n):
		filename = f'{i + initial_seed}_strats.csv'
		file = open(filename, 'r')
		lines = file.readlines()
		file.close()
		start_entry = np.average([float(value.split(',')[0].strip()) for value in lines[:5]])
		end_entry = np.average([float(value.split(',')[0].strip()) for value in lines[-5:]])
		pps_entry = pd.DataFrame({'start': [start_entry], 'end': [end_entry]})
		
		pps = pd.concat([pps, pps_entry], ignore_index=True)
		
	run_stat_tests(pps)


# run n instances of market session. Collect the final profits for each trader for each session and store in a numpy array. Apply the appropriate hypothesis test
def run_experiment(n, market_args, initial_seed):
	# create an empty dataframe
	# traders = market_args[3]
	# buyers = traders['buyers']
	# sellers = traders['sellers']
	# algo_names = list({algo for (algo, _) in buyers+sellers})
	# profits = pd.DataFrame(columns=algo_names)
	profits = None
	ave_profits = pd.DataFrame()
	with Pool(os.cpu_count()) as p:
		print('hello')
		p.map(run_sessions, [(market_args, initial_seed, instance) for instance in range(n)] )
		p.close()
		p.join()
	for i in range(n):
		filename = f'{i + initial_seed}_avg_balance.csv'
		file = open(filename, 'r')
		final_entry = (file.readlines()[-1]).split(',')
	#print(len(final_entry))
	#print(final_entry)
		file.close()
		profit_entry = {}
		for j in range(0, len(final_entry)-2, 2):
			col_name = final_entry[j].strip()
			#print(col_name)
			col_val = final_entry[j+1]
			#print(col_val)
			profit_entry[col_name] = round(float(col_val.strip()))

		#print(profit_entry)
		entry_df = pd.DataFrame([profit_entry])
		#profits = profits.append(profit_entry, ignore_index=True)
	   
		profits = pd.concat([profits, entry_df], ignore_index=True)
	#profits = ave_profits

	print(len(profits))
	run_stat_tests(profits)
	return profits
	#print(f'n: {n}')
#     for i in range(n):
#         random.seed(initial_seed + i)
#         market_session(*market_args)
#         trial_id = market_args[0]
#         filename = f'{trial_id}_avg_balance.csv'
#         file = open(filename, 'r')
#         final_entry = (file.readlines()[-1]).split(',')
#         #print(len(final_entry))
#         #print(final_entry)
#         file.close()
#         profit_entry = {}
#         for j in range(0, len(final_entry)-2, 2):
#             col_name = final_entry[j].strip()
#             #print(col_name)
#             col_val = final_entry[j+1]
#             #print(col_val)
#             profit_entry[col_name] = round(float(col_val.strip()))

#         #print(profit_entry)
#         entry_df = pd.DataFrame([profit_entry])
#         #profits = profits.append(profit_entry, ignore_index=True)
#         profits = pd.concat([profits, entry_df], ignore_index=True)
