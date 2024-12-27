import pandas as pd
import numpy as np
import os
import sys
import click


from fet.pstats import extract_features, feature_cols

# utilites
##################################################################################################

    
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
#################################################################################################    
def select_direction(x,y,direction):
    """
    Selecting values in list 'x' with 'y' value equal to 'direction' (+-1)
    """
    return x[y == direction]

#############################################################################################
    
def count_IPT(TIMES):
    """
    Function counting inter packets times
    """
    PPI_IPT = (TIMES[1:]-TIMES[:-1])  
    return PPI_IPT
###########################################################################################    
def change_types(df):
    """in data frame columns changes lists to arrays of appropriet types"""
    for col in  df.filter(like='TIME_').columns:  
        df[col] = pd.to_datetime(df[col])


    for col in ['D_PHISTS_SIZES','S_PHISTS_SIZES', 'D_PHISTS_IPT','S_PHISTS_IPT', 'PPI_PKT_LENGTHS','PPI_PKT_DIRECTIONS','PPI_PKT_FLAGS']:  
        df[col] = df[col].apply(lambda x: list_transform(x,np.uint32))


    df['PPI_PKT_TIMES'] = df['PPI_PKT_TIMES'].apply(lambda x: list_transform(x,np.datetime64))
    df['PPI_PKT_LENGTHS'] = df['PPI_PKT_LENGTHS'].apply(lambda x: np.array(x,dtype=np.int64)) 
    df['PPI_PKT_DIRECTIONS'] = df['PPI_PKT_DIRECTIONS'].apply(lambda x: np.array(x,dtype=np.int8))
    df['PPI_PKT_FLAGS'] = df['PPI_PKT_FLAGS'].apply(lambda x: np.array(x,dtype=np.int8))
    
    return df
#####################################################################################################
def clean(df,num_PACKETS = 1):
    """
    Removing irrelevant records: empty or uncomplete histogram, flows with less than num_PACKETS packets 
    """
    indices_to_throw = set()

    # histograms should have 8 columns
    for col in df.filter(like='PHISTS').columns:
        indices_to_throw.update(set(list(df[df[col].apply(lambda x: len(x)) < 8].index)))
        
    # nonzeros PHISTS 
    for col in df.filter(like='PHISTS').columns:
        indices_to_throw.update(set(list(df[df[col].apply(sum) == 0].index)))
   
    indices_to_throw = list(indices_to_throw)
    print('number of deleted in PHISTS:' ,df.loc[indices_to_throw, :].shape[0]) 
    df.drop(indices_to_throw, inplace = True)

    # keeping flows with more that  num_PACKETS  packets ( PACKETS + PACKETS_REV )

    print('less than', num_PACKETS,' packets:', df[(df.PACKETS+df.PACKETS_REV)<num_PACKETS].shape[0])
    df = df[(df.PACKETS+ df.PACKETS_REV) >= num_PACKETS]
    
    df.reset_index(inplace=True, drop=True)
    print('shape after cleaning:' ,df.shape) 
    return  df
##############################################################################################################


@click.command()
@click.option('--path_to_data', default="../../data/VNAT_preprocessed/vnat_dataset.pkl", help='Path to data')
@click.option('--folder_to_save', default="../../data/VNAT_preprocessed/", help='Directory path where the output will be saved')
def preprocess(path_to_data,folder_to_save):
    # load data
    print('loading data')
    df = pd.read_pickle(path_to_data)
    file_name = os.path.basename(path_to_data)
    
    # drop records with empty flags
    rows_to_drop = df[df['PPI_PKT_FLAGS'].map(len) == 2].index
    df.drop(rows_to_drop, inplace = True)

    print('extracting statistical features') 
    
    # statistical features
    df = extract_features(df,inplace=False)

    # unifiing upper case letters
    df.columns = df.columns.str.upper()
    

    print('changing types, cleaning')                                                  
    df = change_types(df)
    df = clean(df,20) # at least 20 packets
    
     # total time
    df['TIME_DIFF'] =  df.apply(lambda x: (x.TIME_LAST - x.TIME_FIRST)/ np.timedelta64(1, 's'), axis=1)
    # converting PPI_PKT_TIMES to relative times from TIME_FIRST
    df['PPI_PKT_TIMES'] = df.apply(lambda x: (x.PPI_PKT_TIMES- np.full(len(x.PPI_PKT_TIMES), x.TIME_FIRST, dtype='datetime64[ns]'))/np.timedelta64(1, 's'),axis =1)  
    # inter packets times (S and D direction not separated)
    df['PPI_PKT_IPT'] = df['PPI_PKT_TIMES'].apply(lambda x: count_IPT(x)) 


    # pad PPI statistics by 0 to length 30
    for col in df.filter(like='PPI').columns:
        df[col] = df[col].apply(lambda x: np.pad(x,(0,30 - len(x)),'constant', constant_values = (0,0)))
        
     # separate S and D directions in PPI_PKT_ TIMES / LENGTHS
    df['S_PPI_TIMES'] = df.apply(lambda x: select_direction(x.PPI_PKT_TIMES, x.PPI_PKT_DIRECTIONS,1), axis=1)
    df['D_PPI_TIMES'] = df.apply(lambda x: select_direction(x.PPI_PKT_TIMES, x.PPI_PKT_DIRECTIONS,-1), axis=1)
 
    df['S_PPI_LENGTHS'] = df.apply(lambda x: select_direction(x.PPI_PKT_LENGTHS, x.PPI_PKT_DIRECTIONS,1), axis=1)
    df['D_PPI_LENGTHS'] = df.apply(lambda x: select_direction(x.PPI_PKT_LENGTHS, x.PPI_PKT_DIRECTIONS,-1), axis=1)
    
    # cleaning  PPI_TIMES and computing IPT
    # find flows with less than 2 records (imposible to count IPT)
    indices_to_throw = set()
    for col in df.filter(like='PPI_TIMES').columns:
        indices_to_throw.update(set(list(df[df[col].apply(len) <= 1].index)))

    indices_to_throw = list(indices_to_throw)
    print('number of deleted in PPI_TIMES:' ,df.loc[indices_to_throw, :].shape[0])
    df.drop(indices_to_throw, inplace = True)
    
     #  IPT for  S/D_PPI_TIMES
    df['S_PPI_IPT'] =  df['S_PPI_TIMES'].apply(lambda x: count_IPT(x)) 
    df['D_PPI_IPT'] =  df['D_PPI_TIMES'].apply(lambda x: count_IPT(x)) 

    
    print('saving') 
    filename =  os.path.join(folder_to_save,f"preprocessed_{file_name}")
    df.to_pickle(filename)
##########################################################################################################################

if __name__ == "__main__":
  
    preprocess()
        
