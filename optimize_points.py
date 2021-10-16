"""
File: optimize_points.py
Author: Matt Collins
Created: 9/17/2021

TODO:
- Make this more modular where I can call it pass in inputs. Need to decide on function vs class
- Could explore a dynamic approach to bucketize games. Ie. evenly distribute the odds across 5 buckets
    - This would make it easier to grid search just the point values
- Optimize strategy over an entire season
    - In theory the strategy could be optimized over historical games, and then just applied to new weeks

NOTE - target points per week: ~34
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from utils import append_list_as_row

# set week
week = 7

# load data
odds = pd.read_csv('odds.csv', usecols=['Line', 'win_percentage'])
week_lines = pd.read_csv(f'./data/week{week}.csv')
week_odds = week_lines.merge(odds, how='left', on='Line')

# print(week_odds.sort_values('win_percentage', ascending=False))

strat_bins = [0.0, 0.6, 0.65, 0.7, 0.8, 1.0] # [0.0, 0.6, 0.7, 0.8, 0.9, 1.0]
strat_bin_values = [1, 3, 5, 7, 10] # [1, 4, 9, 10, 10]

# NOTE - uncomment to print out final strat
week_odds['strat'] = pd.cut(week_odds['win_percentage'], strat_bins, labels=strat_bin_values, ordered=False)
week_odds['strat'] = week_odds['strat'].astype('int')
print(week_odds[['Team', 'strat']])


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


# headers = ['Week','Strategy Bins','Strategy Point Values','Strategy Total Points Wagered','Strategy Times Each Point Wagered','Strategy Mean','Strategy Median','Strategy Std','Strategy RAR']
output_values = [week,strat_bins,strat_bin_values,week_odds.strat.sum(),dict(week_odds.groupby("strat")["Team"].count()),round(strat_mean,1),round(strat_median,1),round(strat_std,1),round(strat_rar,2)]


# NOTE - uncomment to write new trials to file
# append_list_as_row('results.csv', headers)

append_list_as_row('results.csv', output_values)



