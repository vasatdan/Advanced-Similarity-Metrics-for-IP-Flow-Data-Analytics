import pandas as pd
import os
import sys
import fnmatch
import click

@click.command()
@click.option('--source_data_folder', default="../../data/VNAT_raw", help='Raw data folder')
@click.option('--save_name', default="../../data/VNAT_preprocessed/vnat_dataset.pkl", help='Output file path')
def load_and_save_data(source_data_folder, save_name):  
    
    NAMES = os.listdir(source_data_folder)
    csv_files = [name for name in NAMES if fnmatch.fnmatch(name, '*.csv')]
    print(csv_files)

    # Load data to one big dataframe, save as .pkl

    dfs = []
    for csv in csv_files:
        print('-' * 50)
        print('category:', csv)
        file = os.path.join(source_data_folder, csv)
        print(file)
        df_temp = pd.read_csv(file)
        df_temp['category'] = csv.split('.')[0].split('_')[1]
        dfs.append(df_temp)
    df = pd.concat(dfs)
  
    
    df.rename(columns = lambda x: x[x.find(" ")+1:], inplace = True)
    df.drop(columns= '0', inplace = True)
    df = df.reset_index(drop=True)
    print('saving dataframe')
    df.to_pickle(save_name)
    
if __name__ == "__main__":
 
    load_and_save_data()