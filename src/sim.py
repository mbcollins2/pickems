import time
import numpy as np
from numpy.lib import median
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class pickSimulator(object):
    def __init__(self, game_odds:pd.DataFrame):
        self.game_odds = game_odds

    def sim(self, n:int, strategy_bins:list, strategy_point_values:list, verbose:bool=False, plot:bool=False) -> list:
        points = []
        start = time.time()
        self.game_odds['strat'] = pd.cut(self.game_odds['win_percentage'], strategy_bins, labels=strategy_point_values, ordered=False)
        self.game_odds['strat'] = self.game_odds['strat'].astype('int')
        for i in range(10000):
            df = self.game_odds.values
            df = np.column_stack((df, np.random.rand(df.shape[0])))
            df = np.column_stack((df, [1 if x==True else -1 for x in df[:,4] <= df[:,1]]))
            df = np.column_stack((df, df[:,5] * df[:,3]))
            negative_points = sum([x for x in df[:,6] if x < 0]) - -40
            penalty = negative_points*2 if negative_points<0 else 0
            pts = np.sum(df[:,6]) + penalty
            points.append(pts)

        if verbose: print(f'Run time: {time.time()-start}')

        if plot:
            med = round(np.median(points))
            sts = round(len([x for x in points if x < -40])/len(points)*100,1)
            sns.histplot(data=points, binwidth=10)
            plt.axvline(x=med, label= f'Median: {round(med)}', color='grey', linestyle='--')
            plt.axvspan(np.min(points), -40, color='r', alpha=0.1)
            plt.suptitle(f'{n} Simulations', weight='bold')
            plt.title(f'Median: {med}   Sucks to Suck: {sts}%')
            plt.legend()
            plt.savefig(f'artifacts/points_{n}_simulations.png', dpi=300)

        return points


    def trials(self, n:int=5, sim_n:int=10000, verbose:bool=False) -> float:
        means = []
        stds = []
        medians = []
        rars = []
        start = time.time()
        for i in range(n):
            points = self.sim(sim_n=sim_n, verbose=False)
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

