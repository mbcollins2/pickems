import time
import numpy as np
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
            # plt.hist(points)
            sns.histplot(data=points, stat='count')
            plt.title(f'{n} Simulations')
            plt.savefig(f'points_{n}_simulations.png')

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

