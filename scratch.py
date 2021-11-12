from src.cluster import get_clusters
from src.utils import read_data
import pandas as pd
from src.sim import pickSimulator

# set week
week = 11


# read in data
week_odds = read_data(week=week, verbose=False)
strat_bins = get_clusters(week_odds, 5)
strat_bin_values = [2, 2, 9, 10, 10] # [1, 4, 9, 10, 10]

sim = pickSimulator(game_odds=week_odds)
sim.sim(n=10000, strategy_bins=strat_bins, strategy_point_values=strat_bin_values, plot=True, verbose=True)


