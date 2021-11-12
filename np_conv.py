import numpy as np
import pandas as pd
import random
from utils import read_data
from cluster import get_clusters

np.random.seed(42)

df = read_data(week=11)
strat_bins = get_clusters(df, 5)
strat_bin_values = [3, 5, 7, 9, 10] # [1, 4, 9, 10, 10]
df['strat'] = pd.cut(df['win_percentage'], strat_bins, labels=strat_bin_values, ordered=False)
df['strat'] = df['strat'].astype('int')



# df['rand'] = np.random.rand(df.shape[0])
# df['win'] = df.apply(lambda x: 1 if x['rand'] <= x['win_percentage'] else -1, axis=1)
# df['points'] = df['win'] * df['strat']
# negative_points = df.loc[df.points<0]['points'].sum() - -40
# penalty = negative_points*2 if negative_points<0 else 0
# pts = df['points'].sum() + penalty
# print(pts)





df = df.values
# # loop
df = np.column_stack((df, np.random.rand(df.shape[0])))
df = np.column_stack((df, [1 if x==True else -1 for x in df[:,4] <= df[:,1]]))
df = np.column_stack((df, df[:,5] * df[:,3]))
negative_points = sum([x for x in df[:,6] if x < 0]) - -40
penalty = negative_points*2 if negative_points<0 else 0
pts = np.sum(df[:,6]) + penalty

# print(np.sum(df[:,6]))
# print(pts)