# script loads data from 'path_to_data' and makes a sample from categories with 'min_cases_in_class' flows taking 'fraction_of_class' of them. Sample is saved  to  folder_to_save as flows_sample_{min_cases_in_class}_{fraction_of_class}.pkl

import pandas as pd
import os
import sys
from timeit import default_timer as timer
from datetime import timedelta
import click

def conditional_sampling(x, n = 50000, frac = 0.0001, random_state = None):
    size = x.shape
    if size[0] > n:
        return x.sample(frac = frac, replace = False, random_state = random_state)

@click.command()
@click.option('--path_to_data', default="../data/preprocessed/clean_flows_SM.pkl", help= 'Path to  data')
@click.option('--folder_to_save', default="../data/preprocessed/", help='Directory path where the sample will be saved')
@click.option('--min_cases_in_class', default= 10000, help='Minimum number of samples required in a class for it to be eligible for sampling.')
@click.option('--fraction_of_class', default= 0.012, help='Fraction of the eligible class to be included in the sample.')

def make_sample(path_to_data, min_cases_in_class, fraction_of_class, folder_to_save):
    print('loading data')
    start = timer()
    df = pd.read_pickle(path_to_data)
    end = timer()
    print('time to load dataframe:', timedelta(seconds = end - start))
    print(df.shape)

    # categories to draw a sample
    categories = 'Domains_collected'
    min_cases_in_class = int(min_cases_in_class)
    fraction_of_class = float(fraction_of_class)
    random_state = 42

    df_sample = df.groupby(categories, group_keys = False).apply(
        lambda x: conditional_sampling(x, n = min_cases_in_class, frac = fraction_of_class, random_state = random_state)
    ).reset_index(drop = True).copy()

    print('sample shape', df_sample.shape)
    print('saving sample')

    # Include values in the filename
    output_filename = os.path.join(folder_to_save,f"sample_{min_cases_in_class}_{fraction_of_class}.pkl")
    df_sample.to_pickle(output_filename)

    df_sample[categories].value_counts()

if __name__ == "__main__":
    
        make_sample()
