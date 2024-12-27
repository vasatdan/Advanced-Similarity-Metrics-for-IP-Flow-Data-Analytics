# perform train, validation, test split, for each set prepare probability distance features, save separately train, val, and test dataframes
import pandas as pd
import numpy as np
import os
import sys
import click

from importlib import import_module

# import of previous preprocessing step
previous_prep_step = import_module('05_split_and_prob_dist_features_extraction')

################################################################################################################    
def novelty_remove_categories(df_train, df_val, novelty_frac, print_info = True):
    """
    Removes novelty classes from the training and validation data
    """
    # novelty split
    cat_counts = df_train.SM_CATEGORY.value_counts()
    if print_info:
        print('-'*50)
        print(f"Processing novelty_frac = {novelty_frac}")
        print("Categories counts:")
        print(cat_counts.head(3))
        print('-'*20)
        print(cat_counts.tail(3))
        print('-'*20)
    # take known_frac of categories
    last_to_take = int(np.ceil(cat_counts.shape[0]*(1-novelty_frac)))
    known_categories = sorted(list(cat_counts[0:last_to_take].index))
    if print_info:
        print(f"Known categories ({len(known_categories)}):", known_categories)
    novel_categories = sorted(list(cat_counts[~cat_counts.index.isin(known_categories)].index))
    if print_info:
        print(f"Novel categories ({len(novel_categories)}):", novel_categories)
    frac = cat_counts[cat_counts.index.isin(known_categories)].sum()/cat_counts.sum()
    if print_info:
        print("Portion of known categories in data: ", frac)
        print('-'*20)
    # narrow the data
    df_train = df_train[df_train.SM_CATEGORY.isin(known_categories)].copy()
    df_val = df_val[df_val.SM_CATEGORY.isin(known_categories)].copy()
    # return
    return df_train, df_val
        
################################################################################################################    
@click.command()
@click.option('--path_to_data', default="../data/preprocessed/preprocessed_sample_10000_0.012.pkl", help= 'Path to source data from step 04')
@click.option('--folder_to_save', default="../data/preprocessed/", help= 'Directory path where the output will be saved')
def main_run(path_to_data, folder_to_save):   
    """
    Encapsulates the overal procedure
    Split & probability distance features calculation
    """
    # split the data to train, validation and test set according to day
    # for different train, val, test split - change this procedure 
    df_train_orig, df_val_orig, df_test_orig = previous_prep_step.train_val_test_split(path_to_data)
    
    # repeat the following code for 25% and 50% novel classes
    for novelty_frac in [0.25, 0.5]:
        # novelty data removal
        df_train, df_val = novelty_remove_categories(df_train_orig, df_val_orig, novelty_frac)
        # copy test data
        df_test = df_test_orig.copy()
        # print statistics
        print(f"Train shape: {df_train.shape}, Validation shape: {df_val.shape}, Test shape: {df_test.shape}")

        # prepare base name for saving (to be followed by _train, _val, _test)
        source_file_base_name = f"novelty_{novelty_frac}_" + os.path.splitext(os.path.basename(path_to_data))[0]

        # extract the probability distance features and save the resulting data files
        previous_prep_step.extract_features(df_train, df_val, df_test, source_file_base_name, folder_to_save)
        
################################################################################################################    
if __name__ == "__main__":
    # main function call
    main_run()
        
        
