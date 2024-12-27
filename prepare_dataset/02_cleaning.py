
import pandas as pd
import numpy as np
import os
import sys
import click

#utils
#########################################################################################
def select_direction(x,y,direction):
    """
    Selecting values in list 'x' with 'y' value equal to 'direction' (+-1)
    """
    return x[y == direction]

def list_transform(input_text, item_dtype):
    """
    changing types of lists
    """
    # put it back if it is not a string
    if type(input_text) is not str:
        return input_text
    # rise exception if it is not a list 
    if not (input_text.startswith("[") and input_text.endswith("]")):
        raise Exception("Bad format of the list")
    input_text = input_text[1:-1]
    if input_text:
        items = input_text.split("|")
        return np.array(items, dtype=item_dtype)
    else:
        return np.array([], dtype=item_dtype)  
    
def temporary_change_types(df):
    """in data frame columns changes lists to arrays of appropriet types"""
#    For cleaning we need correct types but in 04_preprocessing.py original types are needed for FET.exctract_featurs, that is why we create '*_temp' features

    for col in ['D_PHISTS_SIZES','S_PHISTS_SIZES', 'D_PHISTS_IPT','S_PHISTS_IPT']:  
        df[f"{col}_temp"] = df[col].apply(lambda x: list_transform(x,np.uint32))
        
    df['PPI_PKT_TIMES_temp'] = df['PPI_PKT_TIMES'].apply(lambda x: list_transform(x,np.datetime64))
    df['PPI_PKT_LENGTHS_temp'] = df['PPI_PKT_LENGTHS'].apply(lambda x: list_transform(x,np.int64))  
    df['PPI_PKT_DIRECTIONS_temp'] = df['PPI_PKT_DIRECTIONS'].apply(lambda x: list_transform(x,np.int8))
    
    df['PPI_PKT_TIMES_temp'] = df.apply(lambda x: (x.PPI_PKT_TIMES_temp- np.full(len(x.PPI_PKT_TIMES_temp), x.TIME_FIRST, dtype='datetime64[ns]'))/np.timedelta64(1, 's'),axis =1) 
    
    df['S_PPI_TIMES_temp'] = df.apply(lambda x: select_direction(x.PPI_PKT_TIMES_temp, x.PPI_PKT_DIRECTIONS_temp,1), axis=1)
    df['D_PPI_TIMES_temp'] = df.apply(lambda x: select_direction(x.PPI_PKT_TIMES_temp, x.PPI_PKT_DIRECTIONS_temp,-1), axis=1)
 
    df['S_PPI_LENGTHS_temp'] = df.apply(lambda x: select_direction(x.PPI_PKT_LENGTHS_temp, x.PPI_PKT_DIRECTIONS_temp,1), axis=1)
    df['D_PPI_LENGTHS_temp'] = df.apply(lambda x: select_direction(x.PPI_PKT_LENGTHS_temp, x.PPI_PKT_DIRECTIONS_temp,-1), axis=1)
    
    return df
#######################################################################################################################################################

@click.command()
@click.option('--path_to_data', default="../data/preprocessed/flows_SM.pkl", help= 'Path to  data')
@click.option('--save_name', default="../data/preprocessed/clean_flows_SM.pkl", help='Output file path')

def clean(path_to_data, save_name):
    # load data
    print('loading data')
    df = pd.read_pickle(path_to_data)
    print (df.shape)

    print('changing types')
    df = temporary_change_types(df)
    print('cleaning')
    
    """
    Removing irrelevant records: empty or uncomplete histogram, incomplete handshake, imposible to compute inter arrival times
    """
    indices_to_throw = set()

    # histograms should have 8 columns
    for col in df.filter(like='PHISTS_*_temp').columns:
        indices_to_throw.update(set(list(df[df[col].apply(lambda x: len(x)) < 8].index)))
        
    # nonzeros PHISTS and PSTATS 
    for col in df.filter(like='_temp').columns:  
        indices_to_throw.update(set(list(df[df[col].apply(sum) == 0].index)))
   
    indices_to_throw = list(indices_to_throw)
    print('number of deleted in PHISTS and PSTATS:' ,df.loc[indices_to_throw, :].shape[0]) 

    df.drop(indices_to_throw, inplace = True)
    
    # TLS handshake is complete 
    print('less than ', 3,' packets in PSTATS:', df[df.PPI_PKT_TIMES_temp.apply(len)< 3].shape[0])
    df = df[df.PPI_PKT_TIMES_temp.apply(len) >= 3]
    
    # impossible to compute inter arival time
    indices_to_throw = set()
    for col in df.filter(like='PPI_TIMES_temp').columns:
        indices_to_throw.update(set(list(df[df[col].apply(len) <= 1].index)))

    indices_to_throw = list(indices_to_throw)
    print('number of deleted in PPI_TIMES:' ,df.loc[indices_to_throw, :].shape[0])
    df.drop(indices_to_throw, inplace = True)
        
    df.reset_index(inplace=True, drop=True)
    print('shape after cleaning:' ,df.shape) 
    
    print('saving') 
    df.to_pickle(save_name)



if __name__ == "__main__":
     clean()
    
   
        
