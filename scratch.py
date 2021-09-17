import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# load data
odds = pd.read_csv('odds.csv', usecols=['Line', 'win_percentage'])
week = pd.read_csv('week3.csv')
week_odds = week.merge(odds, how='left', on='Line')




strat = 4

def sim(n=10000, strat=5, verbose=False):
    points = []
    start = time.time()
    for i in range(10000):
        week_odds['rand'] = np.random.rand(week_odds.shape[0])
        week_odds['win'] = week_odds.apply(lambda x: 1 if x['rand'] <= x['win_percentage'] else -1, axis=1)
        week_odds['strat'] = strat
        week_odds['points'] = week_odds['win'] * week_odds['strat']

        negative_points = week_odds.loc[week_odds.points<0]['points'].sum() - -40
        penalty = negative_points*2 if negative_points<0 else 0
        pts = week_odds['points'].sum() + penalty
        points.append(pts)

    if verbose: print(f'Run time: {time.time()-start}')

    return points


points = sim(n=10000, strat=5, verbose=True)



# print(points)

plt.hist(points)
plt.savefig('points_baseline.png')

print(f'Strategy: {strat}')
print(f'Points Mean: {np.mean(points)}')
print(f'Points Median: {np.median(points)}')
print(f'Points Std: {np.std(points)}')
print(f'RAR: {np.mean(points)/np.std(points)}')


# print(week_odds.head())
# print(week_odds['points'].sum())

# print(week_odds.apply(lambda x: 1 if x['rand'] <= x['win_percentage'] else 0, axis=1))



# TODO 
# test ndarrays to improve processing time - currently take 16 seconds for 10k sims
# need to add logic for sucks to suck penalty before summing points
