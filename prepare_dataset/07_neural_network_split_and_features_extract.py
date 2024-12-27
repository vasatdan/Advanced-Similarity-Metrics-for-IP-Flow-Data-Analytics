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
def transform_data(df, print_info = True):
    """
    Transforms the data
    """
    # drop S/D_PPI_
    print("Dropping: ", list(df.filter(like='_PPI_').columns))
    df.drop(columns = df.filter(like='_PPI_').columns, inplace = True)
    # drop ORIG_LINE_INDEX a ORIG_SOURCE_FILE
    drop_cols = ["ORIG_LINE_INDEX","ORIG_SOURCE_FILE"]
    print("Dropping: ", drop_cols)
    df.drop(columns = drop_cols, inplace = True)
    # string columns
    drop_cols = list(df.loc[:,df.iloc[0].apply(lambda x: type(x) == str)].columns)
    print("Dropping str columns: ", drop_cols)
    df.drop(columns = drop_cols, inplace = True)
    # list with variable lengths (Statistical features BWD and FWD
    drop_cols = list(df.loc[:,df.iloc[0].apply(lambda x: type(x) == list)].columns)
    print("Dropping list columns: ", drop_cols)
    df.drop(columns = drop_cols, inplace = True)
    # some special unrelevant
    drop_cols = ['DATETIME', 'TIME_FIRST', 'TIME_LAST', 'DURATION', 'DAY']
    print("Dropping: ", drop_cols)
    df.drop(columns = drop_cols, inplace = True)
    # check the np array features
    for col in df.iloc[0][df.iloc[0].apply(lambda x: (type(x) == list) or (type(x) == np.ndarray))].index:
        print("Checking", col)
        # check consistency
        len_lists = ((df[col].apply(len).min() + df[col].apply(len).max())/2)
        if col.startswith("PPI") and len_lists != 30:
            raise ValueError("The length of PPI have to be 30.")
        if col.startswith("D_PHISTS") and len_lists != 8:
            raise ValueError("The length of histograms have to be 8 bins.")
        if col.startswith("S_PHISTS") and len_lists != 8:
            raise ValueError("The length of histograms have to be 8 bins.")
    # return
    return df
    
        
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
    df_train, df_val, df_test = previous_prep_step.train_val_test_split(path_to_data)
    # print statistics
    print(f"Train shape: {df_train.shape}, Validation shape: {df_val.shape}, Test shape: {df_test.shape}")
    # transform
    for df, name in [(df_train, 'train'), (df_val, 'val'), (df_test, 'test')]:
        print("\n----------------------------")
        print(f"Transforming {name}")
        df = transform_data(df)
    # print result
    pd.set_option('display.max_rows', df.shape[1]+1)
    print("\nOutput columns")
    print(df_train.iloc[0].apply(lambda x: type(x)))
    # prepare base name for saving (to be followed by _train, _val, _test)
    save_file_base_name = f"neural_network_" + os.path.splitext(os.path.basename(path_to_data))[0]
    print(f"\nSaving with base name: {save_file_base_name}")
    # saving
    for df, name in [(df_train, 'train'), (df_val, 'val'), (df_test, 'test')]:
        filename = os.path.join(folder_to_save, f"{save_file_base_name}_{name}.pkl")
        df.to_pickle(filename)
        
################################################################################################################    
if __name__ == "__main__":
    # main function call
    main_run()
        
        
