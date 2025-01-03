import pandas as pd
import numpy as np
import sklearn.metrics as metrics
from timeit import default_timer as timer
from datetime import timedelta
from sklearn.metrics import f1_score

import warnings
warnings.simplefilter('ignore', UserWarning)

def drop_features_permute(Xtrain,Xval,ytrain,yval,score,random_seed):
    np.random.seed(random_seed)
    for feature in np.random.permutation(Xval.columns.tolist()):
        Xtrain_0 = Xtrain.drop(columns = feature)
        Xval_0 = Xval.drop(columns = feature)
        
        knn =  KNeighborsClassifier( n_neighbors = 7,  weights = 'distance',p=0.1) 
        knn.fit(Xtrain_0, ytrain)
        predicted = knn.predict(Xval_0)
        score_d = metrics.accuracy_score(yval, predicted)

        if score_d >= score:
            Xtrain = Xtrain.drop(columns = feature)
            Xval = Xval.drop(columns = feature)
            score = score_d
    return Xtrain, Xval


###############################################################
# features

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


features_CM_best = ['D_PHISTS_SIZES_DIST_CM_6', 'S_PHISTS_SIZES_DIST_CM_6', 'D_PHISTS_IPT_DIST_CM_6',
                        'S_PHISTS_IPT_DIST_CM_6', 'S_PPI_IPT_DIST_CM_2', 'D_PPI_IPT_DIST_CM_2', 'S_PPI_LENGTHS_DIST_CM_2',
                        'D_PPI_LENGTHS_DIST_CM_2']
features_W_best = ['D_PHISTS_SIZES_DIST_W_6', 'S_PHISTS_SIZES_DIST_W_6', 'D_PHISTS_IPT_DIST_W_6',
                        'S_PHISTS_IPT_DIST_W_6', 'S_PPI_IPT_DIST_W_2', 'D_PPI_IPT_DIST_W_2', 'S_PPI_LENGTHS_DIST_W_2',
                        'D_PPI_LENGTHS_DIST_W_2']
features_L1_best = ['D_PHISTS_SIZES_DIST_L1_6', 'S_PHISTS_SIZES_DIST_L1_6', 'D_PHISTS_IPT_DIST_L1_6',
                        'S_PHISTS_IPT_DIST_L1_6', 'S_PPI_IPT_DIST_L1_2', 'D_PPI_IPT_DIST_L1_2', 'S_PPI_LENGTHS_DIST_L1_2',
                        'D_PPI_LENGTHS_DIST_L1_2']
features_CM_all = ['D_PHISTS_SIZES_DIST_CM_1', 'S_PHISTS_SIZES_DIST_CM_1', 'D_PHISTS_IPT_DIST_CM_1',
                        'S_PHISTS_IPT_DIST_CM_1','D_PHISTS_SIZES_DIST_CM_2', 'S_PHISTS_SIZES_DIST_CM_2', 'D_PHISTS_IPT_DIST_CM_2',
                        'S_PHISTS_IPT_DIST_CM_2','D_PHISTS_SIZES_DIST_CM_3', 'S_PHISTS_SIZES_DIST_CM_3', 'D_PHISTS_IPT_DIST_CM_3',
                        'S_PHISTS_IPT_DIST_CM_3','D_PHISTS_SIZES_DIST_CM_4', 'S_PHISTS_SIZES_DIST_CM_4', 'D_PHISTS_IPT_DIST_CM_4',
                        'S_PHISTS_IPT_DIST_CM_4','D_PHISTS_SIZES_DIST_CM_5', 'S_PHISTS_SIZES_DIST_CM_5', 'D_PHISTS_IPT_DIST_CM_5',
                        'S_PHISTS_IPT_DIST_CM_5','D_PHISTS_SIZES_DIST_CM_6', 'S_PHISTS_SIZES_DIST_CM_6', 'D_PHISTS_IPT_DIST_CM_6',
                        'S_PHISTS_IPT_DIST_CM_6', 'S_PPI_IPT_DIST_CM_1', 'D_PPI_IPT_DIST_CM_1', 'S_PPI_LENGTHS_DIST_CM_1',
                        'D_PPI_LENGTHS_DIST_CM_1','S_PPI_IPT_DIST_CM_2', 'D_PPI_IPT_DIST_CM_2', 'S_PPI_LENGTHS_DIST_CM_2',
                        'D_PPI_LENGTHS_DIST_CM_2']
features_L1_all = ['D_PHISTS_SIZES_DIST_L1_1', 'S_PHISTS_SIZES_DIST_L1_1', 'D_PHISTS_IPT_DIST_L1_1',
                        'S_PHISTS_IPT_DIST_L1_1','D_PHISTS_SIZES_DIST_L1_2', 'S_PHISTS_SIZES_DIST_L1_2', 'D_PHISTS_IPT_DIST_L1_2',
                        'S_PHISTS_IPT_DIST_L1_2','D_PHISTS_SIZES_DIST_L1_3', 'S_PHISTS_SIZES_DIST_L1_3', 'D_PHISTS_IPT_DIST_L1_3',
                        'S_PHISTS_IPT_DIST_L1_3','D_PHISTS_SIZES_DIST_L1_4', 'S_PHISTS_SIZES_DIST_L1_4', 'D_PHISTS_IPT_DIST_L1_4',
                        'S_PHISTS_IPT_DIST_L1_4','D_PHISTS_SIZES_DIST_L1_5', 'S_PHISTS_SIZES_DIST_L1_5', 'D_PHISTS_IPT_DIST_L1_5',
                        'S_PHISTS_IPT_DIST_L1_5','D_PHISTS_SIZES_DIST_L1_6', 'S_PHISTS_SIZES_DIST_L1_6', 'D_PHISTS_IPT_DIST_L1_6',
                        'S_PHISTS_IPT_DIST_L1_6', 'S_PPI_IPT_DIST_L1_1', 'D_PPI_IPT_DIST_L1_1', 'S_PPI_LENGTHS_DIST_L1_1',
                        'D_PPI_LENGTHS_DIST_L1_1','S_PPI_IPT_DIST_L1_2', 'D_PPI_IPT_DIST_L1_2', 'S_PPI_LENGTHS_DIST_L1_2',
                        'D_PPI_LENGTHS_DIST_L1_2']



###############################################################################

start = timer()
print("Reading data")

# path to data computed by script 05_metric_features_extraction.py
df_train = pd.read_pickle("data/preprocessed/metric_features_preprocessed_sample_10000_0.012_train.pkl")
df_val = pd.read_pickle("data/preprocessed/metric_features_preprocessed_sample_10000_0.012_val.pkl")
end = timer()
print('time to load dataframes:', timedelta(seconds=end-start))

SCORE =[]
SCORE_reduced = []
F1_reduced = []
REDUCTION = {}
random_seed = 5 # the best results gained for CM_all, the best results for L1_all, CM, L1, W  are gained for random_seed = 42 
names = [ 'CM all', 'L1 all', 'CM best', 'L1 best', 'W best'] 
for i,features in enumerate( [features_CM_all, features_L1_all , features_CM_best, features_L1_best,  features_W_best]): 
    print('-'*50)
    Xtrain = df_train[features_B + features_S + features]  
    ytrain = df_train.SM_CATEGORY
    print(names[i])
    print(f"Train: {Xtrain.shape}, {ytrain.shape}")
    Xval = df_val[features_B + features_S + features] 
    yval = df_val.SM_CATEGORY
    print(f"Validation: {Xval.shape}, {yval.shape}")


    from scipy.spatial.distance import minkowski
    from sklearn.neighbors import KNeighborsClassifier

    start = timer()
    knn = KNeighborsClassifier(n_neighbors = 5, weights = 'distance', p=0.1) 
    knn.fit(Xtrain, ytrain)
    predicted = knn.predict(Xval)
    end = timer()

    score_0= metrics.accuracy_score(yval, predicted)
    print('accuracy:',score_0)
    print('execution time:', timedelta(seconds=end-start))
    SCORE.append(score_0)



    Xtrain_old = Xtrain
    Xval_old = Xval
    score_old = score_0
  
    i = 1
    while i>0:
        Xtrain_new,Xval_new = drop_features_permute(Xtrain_old, Xval_old,ytrain, yval,score_old,random_seed)
      
        knn.fit(Xtrain_new, ytrain)
        predicted = knn.predict(Xval_new)
        score_new = metrics.accuracy_score(yval, predicted)
        print('accuracy new:', score_new)
        macro_f1 = f1_score(yval, predicted, average='macro')
        print('f1 new:', macro_f1)

        i = Xtrain_old.shape[1] - Xtrain_new.shape[1] 

        Xtrain_old = Xtrain_new
        Xval_old = Xval_new
        score_old = score_new
        
    SCORE_reduced.append(score_new)
    F1_reduced.append(macro_f1)
    print('features_reduced = ',Xtrain_new.columns)
    print(Xtrain_new.shape[1])
    
    REDUCTION[names[i]]= Xtrain_new.columns

print(SCORE)
print(SCORE_reduced)

import pickle
filename = f'results/reduced_features_{random_seed}.pickle'
with open(filename, 'wb') as handle:
    pickle.dump((names, REDUCTION, SCORE, SCORE_reduced, F1_reduced), handle, protocol=pickle.HIGHEST_PROTOCOL)




