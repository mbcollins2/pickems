import pandas as pd
from csv import writer

def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)


def read_data(week):
    # read in data
    week_lines = pd.read_csv(f'./data/week{week}.csv')
    
    # check if line or percentage
    if week_lines.iloc[0,1] > 0:
        return week_lines.rename(columns={'Line':'win_percentage'})
    
    else:
        odds = pd.read_csv('odds.csv', usecols=['Line', 'win_percentage'])
        return week_lines.merge(odds, how='left', on='Line')
