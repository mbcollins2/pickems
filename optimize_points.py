"""
File: optimize_points.py
Author: Matt Collins
Created: 9/17/2021

TODO:
- Make this more modular where I can call it pass in inputs
- Optimize strategy over an entire season?
    - In theory the strategy could be optimized over historical games, and then just applied to new weeks
    - Will set up an optimizer, then can optimize over single week or season
- Set up optimization
    - Need to make optimize_points a function that returns the primary value to optimize (probably as a negative)
    - Then set up optimization as a separate script called main where you can run everything from

NOTE - target points per week: ~34
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from src.utils import append_list_as_row, read_data
from src.cluster import get_clusters

week = 11

# read in data
week_odds = read_data(week=week, verbose=False)

strat_bins = get_clusters(week_odds, 5)
strat_bin_values = [2, 2, 9, 10, 10] # [1, 4, 9, 10, 10]

# set up optimization
# bounds = tuple([(1,10) for x in strat_bin_values])


# NOTE - uncomment to print out final strat
# week_odds['strat'] = pd.cut(week_odds['win_percentage'], strat_bins, labels=strat_bin_values, ordered=False)
# week_odds['strat'] = week_odds['strat'].astype('int')
# print(week_odds[['Team', 'strat']])




def sim(n=10000, strat_bins=strat_bins, strat_bin_values=strat_bin_values, verbose=False, plot=False):
    points = []
    start = time.time()
    week_odds['strat'] = pd.cut(week_odds['win_percentage'], strat_bins, labels=strat_bin_values, ordered=False)
    week_odds['strat'] = week_odds['strat'].astype('int')
    for i in range(10000):
        df = week_odds.values
        df = np.column_stack((df, np.random.rand(df.shape[0])))
        df = np.column_stack((df, [1 if x==True else -1 for x in df[:,4] <= df[:,1]]))
        df = np.column_stack((df, df[:,5] * df[:,3]))
        negative_points = sum([x for x in df[:,6] if x < 0]) - -40
        penalty = negative_points*2 if negative_points<0 else 0
        pts = np.sum(df[:,6]) + penalty
        points.append(pts)

    if verbose: print(f'Run time: {time.time()-start}')

    if plot:
        plt.hist(points)
        plt.savefig('points_baseline.png')

    return points


def trails(n=30, verbose=False):
    means = []
    stds = []
    medians = []
    rars = []
    start = time.time()
    for i in range(n):
        points = sim(n=10000, verbose=False)
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


strat_mean, strat_median, strat_std, strat_rar = trails(n=5, verbose=True)


# NOTE - headers
# headers = ['Week','Strategy Bins','Strategy Point Values','Strategy Total Points Wagered','Strategy Times Each Point Wagered','Strategy Mean','Strategy Median','Strategy Std','Strategy RAR']
# append_list_as_row('results.csv', headers)


output_values = [week,strat_bins,strat_bin_values,week_odds.strat.sum(),dict(week_odds.groupby("strat")["Team"].count()),round(strat_mean,1),round(strat_median,1),round(strat_std,1),round(strat_rar,2)]
append_list_as_row('results.csv', output_values)



