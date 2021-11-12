from cluster import get_clusters
import pandas as pd

# set week
week = 11


# week_lines = pd.read_csv(f'./data/week{week}.csv')
# print(week_lines)

# print(week_lines.iloc[0,1])
# print(week_lines.columns[1])

# # load data
# odds = pd.read_csv('odds.csv', usecols=['Line', 'win_percentage'])
# week_lines = pd.read_csv(f'./data/week{week}.csv')
# week_odds = week_lines.merge(odds, how='left', on='Line').sort_values('win_percentage', ascending=False)

# cluster = get_clusters(week_odds, 5)

# week_odds['cluster'] = cluster

# print(week_odds)

# strat_bins = week_odds.groupby('cluster')['win_percentage'].max() + 0.001
# print(sorted(sorted(strat_bins.values)[:-1] + [0.0,1.0]))


