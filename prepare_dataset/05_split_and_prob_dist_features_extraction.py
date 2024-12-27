# perform train, validation, test split, for each set prepare probability distance features, save separately train, val, and test dataframes
import pandas as pd
import numpy as np
import os
import sys
import click

from numba import jit
import sklearn.metrics as metrics
from scipy.spatial.distance import pdist
from scipy.stats import wasserstein_distance


from timeit import default_timer as timer
from datetime import timedelta
import sklearn.metrics as metrics
from scipy.spatial.distance import pdist
from scipy.stats import wasserstein_distance


# utils
#######################################################################################
def mean_distribution_PPI(df_column):
        mean_values = np.trim_zeros(df_column.apply(lambda x: np.pad(x, (0, 30 - len(x)), 'constant', constant_values=(0, 0))).mean(), trim='b')
        mean_values.sort()
        unique_values = np.unique(mean_values)
        probabilities = np.unique(mean_values, return_counts=True)[1] / len(mean_values)
        cumulative_distribution = np.cumsum(probabilities)
        return unique_values, probabilities, cumulative_distribution
######################################################################################################   
def mean_distribution_PHISTS(column):
    pref = column.mean() / column.mean().sum()
    dfref = np.cumsum(pref)
    return pref, dfref
######################################################################################################
def ppi_dist(values,prob,dist_type):
    if values[0] !=0:
        prob_ref = np.append(1,np.zeros(len(prob)))
        prob = np.append(0,prob)
    else:
        prob_ref =  np.append(1,np.zeros(len(prob)-1))
    return pdist(np.vstack((prob_ref,prob)),dist_type)

######################################################################################################
def prepare_PPI_PHISTS(df):
    """   
    Prepare vectors of values, probability distribution functions, and cumulative distributions functions   for PPI and vectors of probability distribution functions and cumulative distributions functions   for PPHISTS
    """
    # vectors of values, probability distribution functions, and cumulative distributions functions   for PPI
    for col in ['S_PPI_IPT','D_PPI_IPT','S_PPI_LENGTHS','D_PPI_LENGTHS']:
        df[col].apply(lambda x: x.sort())
        df[f"{col}_values"] = df[col].apply(lambda x: np.unique(x)) 
        df[f"{col}_P"] = df[col].apply(lambda x: np.unique(x,return_counts=True)[1]/len(x)) # probability distribution 
        df[f"{col}_DF"] = df[f"{col}_P"].apply(lambda x: np.cumsum(x))
        
     # vectors of probability distribution functions and cumulative distributions functions for PPHISTS
    for col in df.filter(like='PHISTS').columns:
        df[f"{col}_P"] = df[col].apply(lambda x: x/sum(x))
        df[f"{col}_DF"] = df[f"{col}_P"].apply(lambda x: np.cumsum(x))
   
    return df

######################################################################################################
# numba functions to compute symmetrized Cramér von Mises distance
@jit(nopython=True)
def CramerVonMises_PHISTS(Px,Py,DFx,DFy):
    """
    Symmetrized Cramer von Mises distance between PHISTS. Px - probability distribution, DFx - cumulative distribution function (have to be prepared from PHISTS)
    """
    d = np.dot((DFx-DFy)**2 ,(Px+Py))/2
    return d

######################################################################################################
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
def set_day(x):
    """
    first 2 days (18 and 19/08/2022) -> day 1,
    third day (20/08/2022) -> day 2
    fourth day (21/08/2022) -> day 3
    """
    if x < 202208200000 :
        return 1
    elif (x >= 202208200000 ) and (x < 202208210000 ):
        return 2
    else:
        return 3
    
######################################################################################################
def train_val_test_split(path_to_data):
    """
    Loads data and splits them according to days (last day -> test, the day before -> validation, first day (in reality 2 days) -> train
    """
    print('loading data')
    df_all = pd.read_pickle(path_to_data)
    
    df_all['DAY'] = df_all.apply(lambda x: set_day(x.DATETIME), axis=1)

   
    df_train = df_all[df_all.DAY == 1].copy()
    df_val = df_all[df_all.DAY == 2].copy()
    df_test = df_all[df_all.DAY == 3].copy()
    print(f"Train shape: {df_train.shape}, Validation shape: {df_val.shape}, Test shape: {df_test.shape}")
    return df_train, df_val, df_test

################################################################################################################    
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

################################################################################################################    
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


################################################################################################################    
def extract_features(df_train, df_val, df_test, source_file_base_name, folder_to_save):
    """
    Main function that calculates the probability distance metrics
    Uses train to fit parameters where needed
    """
    # REFERENCE HISTOGRAMS FOR PHISTS
    # get mean histograms, overall mean histogram for training data
    overall_mean_P, overall_mean_DF, Pref_mean, DFref_mean = get_mean_hist(df_train)
    # Define distributions to use
    distributions = [
        (np.array([1, 0, 0, 0, 0, 0, 0, 0]), np.array([1, 1, 1, 1, 1, 1, 1, 1])),  # all happens in the first bin
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
    
    # for train, val, test data prepare probability distance features
    for (df,name) in ([df_train, 'train'], [df_val, 'val'], [df_test, 'test']):
        print(df.shape)
        print('preparing distibutions')
        df = prepare_PPI_PHISTS(df)

        print('computing all distances to all reference objects for', name, 'data')
    # Histogram features
        # Define the columns to be created
        columns = ['D_PHISTS_SIZES', 'S_PHISTS_SIZES', 'D_PHISTS_IPT', 'S_PHISTS_IPT']
        bin_edges_SIZES = np.array([0, 16, 32, 64, 128, 256, 512, 1024])

        # Iterate over distributions and columns to create new columns in df
        for dist_idx, (pref, dfref) in enumerate(distributions):
            for col_idx, col in enumerate(columns):
                col_name_cm = f'{col}_DIST_CM_{dist_idx + 1}'
                col_name_w = f'{col}_DIST_W_{dist_idx + 1}'
                col_name_L1  = f'{col}_DIST_L1_{dist_idx + 1}'
                col_name_L2  = f'{col}_DIST_L2_{dist_idx + 1}'
                col_name_max = f'{col}_DIST_max_{dist_idx + 1}'
                col_name_cos = f'{col}_DIST_cos_{dist_idx + 1}'
                col_name_JS = f'{col}_DIST_JS_{dist_idx + 1}'
                df[col_name_cm] = df.apply(lambda x: CramerVonMises_PHISTS(pref, x[f'{col}_P'], dfref, x[f'{col}_DF']), axis=1)
                df[col_name_w] = df.apply(lambda x: wasserstein_distance(bin_edges_SIZES,bin_edges_SIZES,x[f'{col}_P'],pref), axis=1)
                df[col_name_L1] = df.apply(lambda x: pdist(np.vstack((pref,x[f'{col}_P'])),'cityblock'), axis=1)
                df[col_name_L2] = df.apply(lambda x: pdist(np.vstack((pref,x[f'{col}_P'])),'euclidean'), axis=1)
                df[col_name_max] = df.apply(lambda x: pdist(np.vstack((pref,x[f'{col}_P'])),'chebyshev'), axis=1)
                df[col_name_cos] = df.apply(lambda x: pdist(np.vstack((pref,x[f'{col}_P'])),'cosine'), axis=1)
                df[col_name_JS] = df.apply(lambda x: pdist(np.vstack((pref,x[f'{col}_P'])),'jensenshannon')** 2, axis=1)


        # Create columns for mean distributions

        for i, col in enumerate(columns):
            df[f'{col}_DIST_W_6'] = df.apply(lambda x: wasserstein_distance(bin_edges_SIZES,bin_edges_SIZES,x[f'{col}_P'],Pref_mean[i]), axis=1)
            df[f'{col}_DIST_CM_6'] = df.apply(lambda x: CramerVonMises_PHISTS(Pref_mean[i], x[f'{col}_P'], DFref_mean[i] ,x[f'{col}_DF']), axis=1)
            df[f'{col}_DIST_L1_6'] = df.apply(lambda x: pdist(np.vstack((Pref_mean[i],x[f'{col}_P'])),'cityblock'), axis=1)
            df[f'{col}_DIST_L2_6'] = df.apply(lambda x: pdist(np.vstack((Pref_mean[i],x[f'{col}_P'])),'euclidean'), axis=1)
            df[f'{col}_DIST_max_6'] = df.apply(lambda x: pdist(np.vstack((Pref_mean[i],x[f'{col}_P'])),'chebyshev'), axis=1)
            df[f'{col}_DIST_cos_6'] = df.apply(lambda x: pdist(np.vstack((Pref_mean[i],x[f'{col}_P'])),'cosine'), axis=1)
            df[f'{col}_DIST_JS_6'] = df.apply(lambda x: pdist(np.vstack((Pref_mean[i],x[f'{col}_P'])),'jensenshannon')** 2, axis=1)

    # SPLT features
        # new columns in dataframe containing Wasserstein, Cramér von Mises, l1,l2,max,  cos, Jensen Shannon distances to reference object
        for i, col in enumerate(['S_PPI_IPT', 'D_PPI_IPT', 'S_PPI_LENGTHS', 'D_PPI_LENGTHS']):
            df[f'{col}_DIST_W_1'] = df.apply(lambda x: wasserstein_distance(x[f'{col}_values'],np.array([0]),x[f'{col}_P']), axis=1)
            df[f'{col}_DIST_L1_1'] = df.apply(lambda x: ppi_dist(x[f'{col}_values'],x[f'{col}_P'],'cityblock'), axis=1)
            df[f'{col}_DIST_L2_1'] = df.apply(lambda x: ppi_dist(x[f'{col}_values'],x[f'{col}_P'],'euclidean'), axis=1)
            df[f'{col}_DIST_max_1'] = df.apply(lambda x: ppi_dist(x[f'{col}_values'],x[f'{col}_P'],'chebyshev'), axis=1)
            df[f'{col}_DIST_cos_1'] = df.apply(lambda x: ppi_dist(x[f'{col}_values'],x[f'{col}_P'],'cosine'), axis=1)
            df[f'{col}_DIST_JS_1'] = df.apply(lambda x: ppi_dist(x[f'{col}_values'],x[f'{col}_P'],'jensenshannon')** 2, axis=1)
            df[f'{col}_DIST_CM_1'] = df.apply(lambda x: CramerVonMises_PPI(valuesref,x[f'{col}_values'],Pref_PPI,x[f'{col}_P'],DFref_PPI,x[f'{col}_DF']), axis=1)


            df[f'{col}_DIST_W_2'] = df.apply(lambda x: wasserstein_distance(x[f'{col}_values'],values_ref_mean[i],x[f'{col}_P'],P_ref_mean[i]), axis=1)
            n= len(P_ref_mean[i])
            df[f'{col}_DIST_L1_2'] = df.apply(lambda x: pdist(np.vstack((P_ref_mean[i],np.pad(x[f'{col}_P'], (0, n - len(x[f'{col}_P'])), 'constant', constant_values=(0, 0)))),'cityblock'), axis=1)
            df[f'{col}_DIST_L2_2'] = df.apply(lambda x: pdist(np.vstack((P_ref_mean[i],np.pad(x[f'{col}_P'], (0, n - len(x[f'{col}_P'])), 'constant', constant_values=(0, 0)))),'euclidean'), axis=1)
            df[f'{col}_DIST_max_2'] = df.apply(lambda x: pdist(np.vstack((P_ref_mean[i],np.pad(x[f'{col}_P'], (0, n - len(x[f'{col}_P'])), 'constant', constant_values=(0, 0)))),'chebyshev'), axis=1)
            df[f'{col}_DIST_cos_2'] = df.apply(lambda x: pdist(np.vstack((P_ref_mean[i],np.pad(x[f'{col}_P'], (0, n - len(x[f'{col}_P'])), 'constant', constant_values=(0, 0)))),'cosine'), axis=1)
            df[f'{col}_DIST_JS_2'] = df.apply(lambda x: pdist(np.vstack((P_ref_mean[i],np.pad(x[f'{col}_P'], (0, n - len(x[f'{col}_P'])), 'constant', constant_values=(0, 0)))),'jensenshannon')** 2, axis=1)
            df[f'{col}_DIST_CM_2'] = df.apply(lambda x: CramerVonMises_PPI(values_ref_mean[i],x[f'{col}_values'],P_ref_mean[i],x[f'{col}_P'],DF_ref_mean[i], x[f'{col}_DF']), axis=1)



        print('saving data prepared for metric')    

        features_B =  ['TIME_DIFF', 'BYTES', 'BYTES_REV', 'PACKETS',
                    'PACKETS_REV']
        features_S = ['BYTES_RATE', 'BYTES_REV_RATE', 'BYTES_TOTAL_RATE',
                   'PACKETS_RATE', 'PACKETS_REV_RATE', 'PACKETS_TOTAL_RATE', 'FIN_COUNT', 'SYN_COUNT', 'RST_COUNT', 'PSH_COUNT',
                   'ACK_COUNT', 'URG_COUNT', 'FIN_RATIO', 'SYN_RATIO', 'RST_RATIO',
                   'PSH_RATIO', 'ACK_RATIO', 'URG_RATIO', 'LENGTHS_MIN', 'LENGTHS_MAX',
                   'LENGTHS_MEAN', 'LENGTHS_STD', 'FWD_LENGTHS_MIN', 'FWD_LENGTHS_MAX',
                   'FWD_LENGTHS_MEAN', 'FWD_LENGTHS_STD', 'BWD_LENGTHS_MIN',
                   'BWD_LENGTHS_MAX', 'BWD_LENGTHS_MEAN', 'BWD_LENGTHS_STD', 'PKT_IAT_MIN',
                   'PKT_IAT_MAX', 'PKT_IAT_MEAN', 'PKT_IAT_STD', 'FWD_PKT_IAT_MIN',
                   'FWD_PKT_IAT_MAX', 'FWD_PKT_IAT_MEAN', 'FWD_PKT_IAT_STD',
                   'BWD_PKT_IAT_MIN', 'BWD_PKT_IAT_MAX', 'BWD_PKT_IAT_MEAN',
                   'BWD_PKT_IAT_STD', 'NORM_PKT_IAT_MEAN', 'NORM_PKT_IAT_STD',
                   'NORM_FWD_PKT_IAT_MEAN', 'NORM_FWD_PKT_IAT_STD',
                   'NORM_BWD_PKT_IAT_MEAN', 'NORM_BWD_PKT_IAT_STD']

        phists = ['D_PHISTS_SIZES_DIST_', 'S_PHISTS_SIZES_DIST_', 'D_PHISTS_IPT_DIST_', 'S_PHISTS_IPT_DIST_']
        ppi = ['S_PPI_IPT_DIST_', 'D_PPI_IPT_DIST_', 'S_PPI_LENGTHS_DIST_', 'D_PPI_LENGTHS_DIST_']
        distances = ['CM', 'W', 'JS', 'L1', 'L2','max', 'cos']

        # Create the lists dynamically

        features_dist_CM_all = [f'{prefix}{distances[0]}_{i}' for i in range(1, 7) for prefix in phists] + \
                              [f'{prefix}{distances[0]}_{i}' for i in range(1, 3) for prefix in ppi]
        features_dist_W_all = [f'{prefix}{distances[1]}_{i}' for i in range(1, 7) for prefix in phists] + \
                              [f'{prefix}{distances[1]}_{i}' for i in range(1, 3) for prefix in ppi]
        features_dist_JS_all = [f'{prefix}{distances[2]}_{i}' for i in range(1, 7) for prefix in phists] + \
                              [f'{prefix}{distances[2]}_{i}' for i in range(1, 3) for prefix in ppi]

        features_dist_L1_all = [f'{prefix}{distances[3]}_{i}' for i in range(1, 7) for prefix in phists] + \
                               [f'{prefix}{distances[3]}_{i}' for i in range(1, 3) for prefix in ppi]
        features_dist_L2_all = [f'{prefix}{distances[4]}_{i}' for i in range(1, 7) for prefix in phists] + \
                               [f'{prefix}{distances[4]}_{i}' for i in range(1, 3) for prefix in ppi]
        features_dist_max_all = [f'{prefix}{distances[5]}_{i}' for i in range(1, 7) for prefix in phists] + \
                               [f'{prefix}{distances[5]}_{i}' for i in range(1, 3) for prefix in ppi]
        features_dist_cos_all = [f'{prefix}{distances[6]}_{i}' for i in range(1, 7) for prefix in phists] + \
                               [f'{prefix}{distances[6]}_{i}' for i in range(1, 3) for prefix in ppi]

        features_dist_all = features_dist_CM_all + features_dist_W_all + features_dist_JS_all + features_dist_L1_all + features_dist_L2_all + features_dist_max_all + features_dist_cos_all


        targets = [ 'DOMAINS_COLLECTED', 'SM_CATEGORY', 'SERVICE','SERVICE GROUP', 'SERVICE CATEGORY']


        filename = os.path.join(folder_to_save,f"metric_features_{source_file_base_name}_{name}.pkl")
        df[features_B + features_S + features_dist_all + targets + ['ORIG_LINE_INDEX', 'ORIG_SOURCE_FILE'] ].to_pickle(filename)
        
################################################################################################################    
@click.command()
@click.option('--path_to_data', default="../data/preprocessed/preprocessed_sample_10000_0.012.pkl", help= 'Path to source data')
@click.option('--folder_to_save', default="../data/preprocessed/", help= 'Directory path where the output will be saved')
def main_run(path_to_data, folder_to_save):   
    """
    Encapsulates the overal procedure
    Split & probability distance features calculation
    """
    # split the data to train, validation and test set according to day
    # for different train, val, test split - change this procedure 
    df_train, df_val, df_test = train_val_test_split(path_to_data)
    
    # prepare base name for saving (to be followed by _train, _val, _test)
    source_file_base_name = os.path.splitext(os.path.basename(path_to_data))[0]
    
    # extract the probability distance features and save the resulting data files
    extract_features(df_train, df_val, df_test, source_file_base_name, folder_to_save)
    
        
################################################################################################################    
if __name__ == "__main__":
    # main function call
    main_run()
        
        
