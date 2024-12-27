import pandas as pd
import numpy as np
import os
import sys
import click
# statistical features extraction packet
from fet.pstats import extract_features, feature_cols
# utils
###########################################################################
def modify_types(df):
    """in data frame columns changes lists to arrays of appropriet types"""
   
    for col in  df.filter(like='TIME_').columns:  
        df[col] = pd.to_datetime(df[col])
   
   
    df['DATETIME'] = df['DATETIME'].apply(lambda x: int(x))
    df['PPI_PKT_FLAGS'] = df['PPI_PKT_FLAGS'].apply(lambda x: np.array(x,dtype=np.int8))
    
    # Get the list of columns with '_temp'
    columns_with_temp = [col for col in df.columns if '_TEMP' in col]
    # Remove '_temp' from the column names and create a list of original column names
    original_columns = [col.replace('_TEMP', '') for col in columns_with_temp]

    columns_to_drop = [value for value in original_columns if value in df.columns]
    # Drop the original columns
    df.drop(columns= columns_to_drop, inplace=True)
   
    # Rename the columns with '_temp' to remove '_temp'
    rename_dict = {col: col.replace('_TEMP', '') for col in columns_with_temp}
    df.rename(columns = rename_dict, inplace=True)
    return df

           
def count_IPT(TIMES):
    """
    Function counting inter packets times
    """
    PPI_IPT = (TIMES[1:]-TIMES[:-1])
    return PPI_IPT
############################################################################################################š

@click.command()
@click.option('--path_to_data', default="../data/preprocessed/sample_10000_0.012.pkl", help= 'Path to  data')
@click.option('--folder_to_save', default="../data/preprocessed/", help='Directory path where the sample will be saved')

def preprocess(path_to_data,folder_to_save):
    # load data
    print('loading data')
    df = pd.read_pickle(path_to_data)
    file_name = os.path.basename(path_to_data)
        
    print('extracting staistical features') 
    df = extract_features(df,inplace=False)

    # unifiing upper case letters
    df.columns = df.columns.str.upper()
    
    print('changing types')                                                  
    df = modify_types(df)
    df['TIME_DIFF'] =  df.apply(lambda x: (x.TIME_LAST - x.TIME_FIRST)/ np.timedelta64(1, 's'), axis=1)
    df['PPI_PKT_IPT'] = df['PPI_PKT_TIMES'].apply(lambda x: count_IPT(x)) 

    # pad PPI statistics by 0 to length 30
    for col in df.filter(like='PPI_PKT').columns:
        df[col] = df[col].apply(lambda x: np.pad(x,(0,30 - len(x)),'constant', constant_values = (0,0)))
    
     #  IPT for  S/D_PPI_TIMES
    df['S_PPI_IPT'] =  df['S_PPI_TIMES'].apply(lambda x: count_IPT(x)) 
    df['D_PPI_IPT'] =  df['D_PPI_TIMES'].apply(lambda x: count_IPT(x)) 
    

    S_feature_cols_upper = [x.upper() for x in feature_cols + ['BWD','FWD']]
    cols_without_S = [item for item in df.columns if item not in S_feature_cols_upper]
       
    print('saving') 
    filename = os.path.join(folder_to_save,f"preprocessed_{file_name}")
    df.to_pickle(filename)
    
    # preprocessed data, only original columns
    df_nn = df[cols_without_S]
    filename2 = os.path.join(folder_to_save,f"preprocessed_without_S_{file_name}")
    df_nn.to_pickle(filename2)


if __name__ == "__main__":
        preprocess()
        
