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

def anova(df):
	anova = pg.rm_anova(data=df, correction=True)
	sphericity = anova['p-spher'][0]
	pvalue = None
	if anova['sphericity'][0]:
		pvalue = anova['p-unc'][0]
	else:
		pvalue = anova['p-GG-corr'][0]

	return pvalue

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

def run_stats(df : pd.DataFrame, summary_entry):
	is_normal = True
	for col in df.columns:
		_, pvalue = stats.shapiro(df[col])
		if pvalue < 0.05:
			is_normal = False
		summary_entry[f'{col} normality'] = pvalue

	summary_entry['All normal'] = is_normal
	# check if comes from the same population
	columns = [df[column] for column in df.columns]
	is_two_dist = len(columns) == 2
	pvalue = None
	further_test = False
	test_name = None
	if is_normal and is_two_dist:
		test_name = 'paired t-test'
		_, pvalue = stats.ttest_rel(*columns)
	elif is_normal and (not is_two_dist):
		test_name = 'Repeated Measures Anova'
		pvalue = anova(df)
		further_test = pvalue < 0.05
	elif (not is_normal) and is_two_dist:
		test_name = 'Wilcoxon Signed Rank test'
		_, pvalue = stats.wilcoxon(*columns)
	else:
		test_name = 'Friedman Test'
		_, pvalue = stats.friedmanchisquare(*columns)
		further_test = pvalue < 0.05

	summary_entry['Test Name'] = test_name
	summary_entry['P Value'] = pvalue
	summary_entry['Different populations'] = pvalue < 0.05

	return summary_entry


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

def create_threads(market_args, initial_seed, n):
	profits = pd.DataFrame()
	print(market_args)
	with Pool(os.cpu_count()) as p:
		p.map(run_sessions, [(market_args, initial_seed, instance) for instance in range(n)] )
		p.close()
		p.join()
	for i in range(n):
		filename = f'{i + initial_seed}_avg_balance.csv'
		file = open(filename, 'r')
		final_entry = (file.readlines()[-1]).split(',')
		file.close()
		profit_entry = {}
		for j in range(0, len(final_entry)-2, 2):
			col_name = final_entry[j].strip()
			col_val = final_entry[j+1]
			profit_entry[col_name] = float(col_val.strip())
		entry_df = pd.DataFrame([profit_entry])
		profits = pd.concat([profits, entry_df], ignore_index=True)
	return profits

def part_a(n, market_args, initial_seed, r_vals):
	stat_summary = pd.DataFrame()
	fig, axes = plt.subplots(2,2)
	for i in range(len(n)):
		profits = create_threads(market_args, initial_seed, n[i])
		summary_entry = {'n' : n[i], 'Trader ratio:' : ':'.join(str(r) for r in r_vals)}
		summary_entry = run_stats(profits, summary_entry)
		summmary_entry = pd.DataFrame([summary_entry])
		stat_summary = pd.concat([stat_summary, summmary_entry], ignore_index=True)
		sns.kdeplot(data=profits, fill=False, ax=axes[i, 0])
		sns.boxplot(data=profits, ax=axes[i, 1])
	return stat_summary

def part_b(n, market_args, r_list, algos, initial_seed):
	stat_summary = pd.DataFrame()
	fig, axes = plt.subplots(1,len(r_list))
	for i in range(len(r_list)):
		r_vals = (r_list[i], 100 - r_list[i])
		buyer_specs = get_traders_specs(algos, r_vals, 20)
		seller_specs = get_traders_specs(algos, r_vals, 20)
		market_args[2] = {'sellers': seller_specs, 'buyers': buyer_specs}
		profits = create_threads(market_args, initial_seed, n)
		summary_entry = {'Trader ratio' : ':'.join(str(r) for r in r_vals)}
		summary_entry = run_stats(profits, summary_entry)
		summary_entry = pd.DataFrame([summary_entry])
		stat_summary = pd.concat([stat_summary, summary_entry], ignore_index=True)
		sns.violinplot(data=profits, inner='box', cut = 0, ax=axes[i])
	return stat_summary
	
def part_c(n, market_args, r_vals, algos, initial_seed):
	stat_summary = pd.DataFrame()
	fig, axes =  None, None
	if len(set(r_vals)) != 1:
		fig, axes = plt.subplots(1, len(r_vals))
	print(get_traders_specs(algos, r_vals, 20))
	for i in range(len(r_vals)):
		buyer_specs = get_traders_specs(algos, r_vals, 20)
		seller_specs = get_traders_specs(algos, r_vals, 20)
		market_args[2] = {'sellers': seller_specs, 'buyers': buyer_specs}
		profits = create_threads(market_args, initial_seed, n)
		summary_entry = {'Trader ratio' : ':'.join(str(r) for r in r_vals)}
		print(profits.head())
		print(profits.tail())
		summary_entry = run_stats(profits, summary_entry)
		summary_entry = pd.DataFrame([summary_entry])
		stat_summary = pd.concat([stat_summary, summary_entry], ignore_index=True)
		if len(set(r_vals)) == 1: # no need to permute if all elements are equal
			sns.violinplot(data=profits, inner='box')
			return stat_summary
		else:
			r_vals.append(r_vals.pop(0)) # rotates by 1
			sns.violinplot(data=profits, inner='box', cut = 0, ax=axes[i])
	return stat_summary
			

def part_d1(n, market_args, initial_seed):
	stat_summary = pd.DataFrame()
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
		start_entry = np.average([float(value.split(',')[15].strip()) for value in lines[:5]])
		end_entry = np.average([float(value.split(',')[15].strip()) for value in lines[-5:]])
		pps_entry = pd.DataFrame({'inital pps': [start_entry], 'final pps': [end_entry]})
		
		pps = pd.concat([pps, pps_entry], ignore_index=True)
	
	summary_entry = run_stats(pps_entry, {})
	summary_entry = pd.DataFrame([summary_entry])
	stat_summary = pd.concat([stat_summary, summary_entry], ignore_index=True)
	fig, axes = plt.subplots(1,2)
	sns.kdeplot(data=pps, fill=False, ax=axes[0])
	sns.boxplot(data=pps, ax=axes[1])
	return stat_summary


# run n instances of market session. Collect the final profits for each trader for each session and store in a numpy array. Apply the appropriate hypothesis test
def run_experiment_old(n, market_args, initial_seed):
	# create an empty dataframe
	# traders = market_args[3]
	# buyers = traders['buyers']
	# sellers = traders['sellers']
	# algo_names = list({algo for (algo, _) in buyers+sellers})
	# profits = pd.DataFrame(columns=algo_names)
	profits = None
	# ave_profits = pd.DataFrame()
	# with Pool(os.cpu_count()) as p:
	# 	print('hello')
	# 	p.map(run_sessions, [(market_args, initial_seed, instance) for instance in range(n)] )
	# 	p.close()
	# 	p.join()
	# for i in range(n):
	# 	filename = f'{i + initial_seed}_avg_balance.csv'
	# 	file = open(filename, 'r')
	# 	final_entry = (file.readlines()[-1]).split(',')
	# #print(len(final_entry))
	# #print(final_entry)
	# 	file.close()
	# 	profit_entry = {}
	# 	for j in range(0, len(final_entry)-2, 2):
	# 		col_name = final_entry[j].strip()
	# 		#print(col_name)
	# 		col_val = final_entry[j+1]
	# 		#print(col_val)
	# 		profit_entry[col_name] = round(float(col_val.strip()))

	# 	#print(profit_entry)
	# 	entry_df = pd.DataFrame([profit_entry])
	# 	#profits = profits.append(profit_entry, ignore_index=True)
	   
	# 	profits = pd.concat([profits, entry_df], ignore_index=True)
	# #profits = ave_profits

	# print(len(profits))
	# run_stat_tests(profits)
	# return profits
	print('hello')
	print(f'n: {n}')
	for i in range(n):
		random.seed(initial_seed + i)
		market_session(*market_args)
		trial_id = market_args[0]
		filename = f'{trial_id}_avg_balance.csv'
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
	run_stat_tests(profits)
