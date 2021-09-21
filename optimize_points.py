import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from utils import append_list_as_row

# target points per week: ~34

# load data
odds = pd.read_csv('odds.csv', usecols=['Line', 'win_percentage'])
week = pd.read_csv('week3.csv')
week_odds = week.merge(odds, how='left', on='Line')

# set week
week = 3


strat_bins = [.000, .600, .700, .800, .900, 1.000]
strat_bin_values = [1, 4, 9, 10, 10] # current best: [1, 4, 9, 10, 10]

strat = 4

def sim(n=10000, strat=5, strat_bins=strat_bins, strat_bin_values=strat_bin_values, verbose=False, plot=False):
    points = []
    start = time.time()
    for i in range(10000):
        week_odds['rand'] = np.random.rand(week_odds.shape[0])
        week_odds['win'] = week_odds.apply(lambda x: 1 if x['rand'] <= x['win_percentage'] else -1, axis=1)
        week_odds['strat'] = pd.cut(week_odds['win_percentage'], strat_bins, labels=strat_bin_values, ordered=False)
        week_odds['strat'] = week_odds['strat'].astype('int')
        # week_odds['strat'] = strat
        week_odds['points'] = week_odds['win'] * week_odds['strat']

        negative_points = week_odds.loc[week_odds.points<0]['points'].sum() - -40
        penalty = negative_points*2 if negative_points<0 else 0
        pts = week_odds['points'].sum() + penalty
        points.append(pts)

    if verbose: print(f'Run time: {time.time()-start}')

    if plot:
        plt.hist(points)
        plt.savefig('points_baseline.png')

    return points


def bootstrap(n=30, verbose=False):
    means = []
    stds = []
    medians = []
    rars = []
    start = time.time()
    for i in range(n):
        points = sim(n=10000, strat=strat, verbose=False)
        means.append(np.mean(points))
        medians.append(np.median(points))
        stds.append(np.std(points))
        rars.append(np.mean(points)/np.std(points))
    
    if verbose: print(f'Run time: {time.time()-start}')

    strat_mean = np.mean(means)
    strat_median = np.mean(medians)
    strat_std = np.mean(stds)
    strat_rar = np.mean(rars)

    return strat_mean, strat_median, strat_std, strat_rar 




strat_mean, strat_median, strat_std, strat_rar = bootstrap(n=3, verbose=False)

week_odds['strat'] = pd.cut(week_odds['win_percentage'], strat_bins, labels=strat_bin_values, ordered=False)
week_odds['strat'] = week_odds['strat'].astype('int')
# print(week_odds[['Team', 'strat']])

# print(f'Strategy Bins: {strat_bins}', file=open('output.txt', 'a'))
# print(f'Strategy Point Values: {strat_bin_values}', file=open('output.txt', 'a'))
# print(f'Strategy Total Points Wagered: {week_odds.strat.sum()}', file=open('output.txt', 'a'))
# print(f'Strategy Times Each Point Wagered: {dict(week_odds.groupby("strat")["Team"].count())}', file=open('output.txt', 'a'))

# print(f'Strategy Mean: {round(strat_mean,1)}', file=open('output.txt', 'a'))
# print(f'Strategy Median: {round(strat_median,1)}', file=open('output.txt', 'a'))
# print(f'Strategy Std: {round(strat_std,1)}', file=open('output.txt', 'a'))
# print(f'Strategy RAR: {round(strat_rar,1)}', file=open('output.txt', 'a'))

# print(f'\n', file=open('output.txt', 'a'))
# print(f'-------------------------------------------------------------', file=open('output.txt', 'a'))
# print(f'\n', file=open('output.txt', 'a'))

headers = ['Week','Strategy Bins','Strategy Point Values','Strategy Total Points Wagered','Strategy Times Each Point Wagered','Strategy Mean','Strategy Median','Strategy Std','Strategy RAR']
output_values = [week,strat_bins,strat_bin_values,week_odds.strat.sum(),dict(week_odds.groupby("strat")["Team"].count()),round(strat_mean,1),round(strat_median,1),round(strat_std,1),round(strat_rar,1)]

# append_list_as_row('results.csv', headers)
# append_list_as_row('results.csv', output_values)

# TODO 
# test ndarrays to improve processing time - currently take 20 seconds for 10k sims
# update to write results to a table so it's easier to filter and sort