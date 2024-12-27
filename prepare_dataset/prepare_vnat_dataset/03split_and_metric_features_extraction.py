import pandas as pd
import numpy as np
import os
import sys
import click

from numba import jit
import sklearn.metrics as metrics
# from scipy.spatial.distance import pdist
# from scipy.stats import wasserstein_distance
from sklearn.model_selection import train_test_split

# utils
########################################################################################################################################
def mean_distribution_PPI(df_column):
        mean_values = np.trim_zeros(df_column.apply(lambda x: np.pad(x, (0, 30 - len(x)), 'constant', constant_values=(0, 0))).mean(), trim='b')
        mean_values.sort()
        unique_values = np.unique(mean_values)
        probabilities = np.unique(mean_values, return_counts=True)[1] / len(mean_values)
        cumulative_distribution = np.cumsum(probabilities)
        return unique_values, probabilities, cumulative_distribution
#############################################################################################    
def mean_distribution_PHISTS(column):
    pref = column.mean() / column.mean().sum()
    dfref = np.cumsum(pref)
    return pref, dfref
#########################################################################################
def prepare_PPI_PHISTS(df):
    """   
    Prepare vectors of values, probability distribution functions, and cumulative distributions functions   for PPI and vectors of probability distribution functions and cumulative distributions functions   for PPHISTS
    """
    # vectors of values, probability distribution functions, and cumulative distributions functions   for PPI
    for col in ['S_PPI_IPT','D_PPI_IPT','S_PPI_LENGTHS','D_PPI_LENGTHS']:
        df[col].apply(lambda x: x.sort())
        df[f"{col}_values"] = df[col].apply(lambda x: np.unique(x)) 
        df[f"{col}_P"] = df[col].apply(lambda x: np.unique(x,return_counts=True)[1]/len(x)) 
        df[f"{col}_DF"] = df[f"{col}_P"].apply(lambda x: np.cumsum(x))
        
     # vectors of probability distribution functions and cumulative distributions functions   for PPHISTS
    for col in df.filter(like='PHISTS').columns:
        df[f"{col}_P"] = df[col].apply(lambda x: x/sum(x))
        df[f"{col}_DF"] = df[f"{col}_P"].apply(lambda x: np.cumsum(x))
   
    return df
##########################################################################################
def get_mean_hist(df_train):
     # Create the mean distributions for each column of train data
    Pref_DPD, DFref_DPD = mean_distribution_PHISTS(df_train.D_PHISTS_SIZES)
    Pref_SPD, DFref_SPD = mean_distribution_PHISTS(df_train.S_PHISTS_SIZES)
    Pref_DPI, DFref_DPI = mean_distribution_PHISTS(df_train.D_PHISTS_IPT)
    Pref_SPI, DFref_SPI = mean_distribution_PHISTS(df_train.S_PHISTS_IPT)

    # Overall mean of train data
    overall_mean_P = np.mean([Pref_DPD,Pref_SPD,Pref_DPI,Pref_SPI], axis=0)
    overall_mean_DF = np.cumsum(overall_mean_P)
    
    Pref_mean = [Pref_DPD, Pref_SPD, Pref_DPI, Pref_SPI]
    DFref_mean = [DFref_DPD, DFref_SPD, DFref_DPI, DFref_SPI]
    
    return overall_mean_P, overall_mean_DF, Pref_mean, DFref_mean
############################################################################################
def get_mean_splt(df_train):
    # averadge values on i-th positions of S/D_ IPT and LENGHTS for each column of train data
    S_PPI_IPT_mean_values, S_PPI_IPT_mean_P, S_PPI_IPT_mean_DF = mean_distribution_PPI(df_train.S_PPI_IPT)
    D_PPI_IPT_mean_values, D_PPI_IPT_mean_P, D_PPI_IPT_mean_DF = mean_distribution_PPI(df_train.D_PPI_IPT)
    S_PPI_LENGTHS_mean_values, S_PPI_LENGTHS_mean_P, S_PPI_LENGTHS_mean_DF = mean_distribution_PPI(df_train.S_PPI_LENGTHS)
    D_PPI_LENGTHS_mean_values, D_PPI_LENGTHS_mean_P, D_PPI_LENGTHS_mean_DF = mean_distribution_PPI(df_train.D_PPI_LENGTHS)
    
    DF_ref_mean = [S_PPI_IPT_mean_DF, D_PPI_IPT_mean_DF, S_PPI_LENGTHS_mean_DF, D_PPI_LENGTHS_mean_DF]
    P_ref_mean = [S_PPI_IPT_mean_P, D_PPI_IPT_mean_P, S_PPI_LENGTHS_mean_P, D_PPI_LENGTHS_mean_P]
    values_ref_mean = [S_PPI_IPT_mean_values,D_PPI_IPT_mean_values,S_PPI_LENGTHS_mean_values,D_PPI_LENGTHS_mean_values]
    
    return P_ref_mean, DF_ref_mean, values_ref_mean 

# numba functions to compute symmetrized Cramér von Mises distance
###############################################################################################
@jit(nopython=True)
def CramerVonMises_PHISTS(Px,Py,DFx,DFy):
    """
    Symmetrized Cramer von Mises distance between PHISTS. Px - probability distribution, DFx - cumulative distribution function (have to be prepared from PHISTS)
    """
    d = np.dot((DFx-DFy)**2 ,(Px+Py))/2
    return d

@jit(nopython=True)    
def CramerVonMises_PPI(valuesx,valuesy,Px,Py,DFx,DFy):
    """
    Symmetrized Cramer von Mises distance between records in  PPI (first 30 packets). valuesx - increasingly ordered unique values from PPI, Px -probability distribution of valuesx, DFx - cumulative distribution function (have to be prepared from PPI)
    """
    
    
    nx = len(valuesx);  ny = len(valuesy)

    diffx =np.zeros(nx,np.float64)
    diffy =np.zeros(ny,np.float64)
    
    for i,value in enumerate(valuesx):
        diffx[i] = (np.where(valuesy<=value, Py,0).sum() - DFx[i])**2 

    for i,value in enumerate(valuesy):
        diffy[i] = (np.where(valuesx<=value, Px,0).sum() - DFy[i])**2 
    dist = (np.dot(diffx,Px) + np.dot(diffy,Py))/2  
    return dist   
######################################################################################################

def extract_features(df_train, df_test, source_file_base_name, folder_to_save):
     
        
 # REFERENCE HISTOGRAMS FOR PHISTS
 # get mean histograms, overall mean histogram  for training data
    overall_mean_P, overall_mean_DF, Pref_mean, DFref_mean = get_mean_hist(df_train)
    # Define distributions to use
    distributions = [
        (np.array([1, 0, 0, 0, 0, 0, 0, 0]), np.array([1, 1, 1, 1, 1, 1, 1, 1])),    # all happens in the first bin
        (np.array([0, 0, 0, 0, 0, 0, 0, 1]), np.array([0, 0, 0, 0, 0, 0, 0, 1])),  # all happene in the last bin
        (np.array([1/8]*8), np.array([1/8, 2/8, 3/8, 4/8, 5/8, 6/8, 7/8, 1])),     # the same probability of the each bin (corresponds to geometric ditribution of the data)
        (np.array([1/128, 1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2]), np.cumsum([1/128, 1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2])),   # geometric increase of bin probability (coresponds to uniform distribution of the data)
        (overall_mean_P, overall_mean_DF)                                          # overall mean over all PHISTS
                    ]
    
# REFERENCE VALUES, P AND DF FUNCTIONS FOR PPI_PKT 
    # only one jump in 0 value
    valuesref = np.zeros(1)
    Pref_PPI = np.ones(1)
    DFref_PPI = np.ones(1)
    # get  mean splts for training data
    P_ref_mean, DF_ref_mean, values_ref_mean = get_mean_splt(df_train)
    
    # for train, test data prepare probability distance features
    for (df,name) in ([df_train, 'train'],  [df_test, 'test']):
        print(df.shape)
        print('preparing distibutions')
        df = prepare_PPI_PHISTS(df)

        print('computing distances to all reference objects for', name, 'data')
   # Histogram features
        # Define the columns to be created
        columns = ['D_PHISTS_SIZES', 'S_PHISTS_SIZES', 'D_PHISTS_IPT', 'S_PHISTS_IPT']
        bin_edges_SIZES = np.array([0, 16, 32, 64, 128, 256, 512, 1024])

        # Iterate over distributions and columns to create new columns in df
        for dist_idx, (pref, dfref) in enumerate(distributions):
            for col_idx, col in enumerate(columns):
                col_name = f'{col}_DIST_CM_{dist_idx + 1}'
                df[col_name] = df.apply(lambda x: CramerVonMises_PHISTS(pref, x[f'{col}_P'], dfref, x[f'{col}_DF']), axis=1)


        for i, col in enumerate(columns):
            df[f'{col}_DIST_CM_6'] = df.apply(lambda x: CramerVonMises_PHISTS(Pref_mean[i], x[f'{col}_P'], DFref_mean[i] ,x[f'{col}_DF']), axis=1)


    # SPLT features
        # new columns in dataframe containing CM distance to reference object
        for i, col in enumerate(['S_PPI_IPT', 'D_PPI_IPT', 'S_PPI_LENGTHS', 'D_PPI_LENGTHS']):

            df[f'{col}_DIST_CM_1'] = df.apply(lambda x: CramerVonMises_PPI(valuesref,x[f'{col}_values'],Pref_PPI,x[f'{col}_P'],DFref_PPI,x[f'{col}_DF']), axis=1)

            df[f'{col}_DIST_CM_2'] = df.apply(lambda x: CramerVonMises_PPI(values_ref_mean[i],x[f'{col}_values'],P_ref_mean[i],x[f'{col}_P'],DF_ref_mean[i], x[f'{col}_DF']), axis=1)


        print('saving data prepared for metric')
        filename = os.path.join(folder_to_save,f"metric_features_{source_file_base_name}_{name}.pkl")
        df.to_pickle(filename)
###############################################################################################################################
@click.command()
@click.option('--path_to_data', default="../../data/VNAT_preprocessed/preprocessed_vnat_dataset.pkl", help='Path to data')
@click.option('--folder_to_save', default="../../data/VNAT_preprocessed/", help='Directory path where the output will be saved')
def main_run(path_to_data, folder_to_save):   
    """
    Encapsulates the overal procedure
    Split & probability distance features calculation
    """
    df = pd.read_pickle(path_to_data)
    # split the data to train and test set according to day
    # for different train,  test split change this procedure 
    df_train,  df_test = train_test_split(df, test_size = 0.2, random_state = 42, stratify = df.LABEL)
    
    # prepare base name for saving (to be followed by _train, _val, _test)
    source_file_base_name = os.path.splitext(os.path.basename(path_to_data))[0]
    
    # extract the probability distance features and save the resulting data files
    extract_features(df_train, df_test, source_file_base_name, folder_to_save)
    
################################################################################################################    
if __name__ == "__main__":
    # main function call
    main_run()