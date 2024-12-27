import pandas as pd
import numpy as np
import os
import sys
from timeit import default_timer as timer
from datetime import timedelta
import sklearn.metrics as metrics
from scipy.spatial.distance import pdist
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import minkowski
from sklearn.neighbors import KNeighborsClassifier

import warnings
warnings.simplefilter('ignore', UserWarning)

# path to data computed by script 05_metric_features_extraction.py
print("Reading data")
df_train = pd.read_pickle('data/preprocessed/metric_features_preprocessed_sample_10000_0.012_train.pkl')
df_val = pd.read_pickle('data/preprocessed/metric_features_preprocessed_sample_10000_0.012_val.pkl')


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



PHISTS =  ['D_PHISTS_SIZES_DIST', 'S_PHISTS_SIZES_DIST', 'D_PHISTS_IPT_DIST', 'S_PHISTS_IPT_DIST']
PPI = ['S_PPI_IPT_DIST', 'D_PPI_IPT_DIST', 'S_PPI_LENGTHS_DIST', 'D_PPI_LENGTHS_DIST']

FEATURES_DIST = {}
for app in ['L1','L2','max','cos','JS', 'W', 'CM']:
    temp_PHISTS_all = []
    for ap1 in ['1','2','3','4','5','6']:
        temp_PHISTS = [name +'_'+ app + '_' + ap1 for name in PHISTS]
        temp_PHISTS_all = temp_PHISTS_all + temp_PHISTS
        for ap2 in ['1','2']:
            temp_PPI = [name +'_'+ app + '_' + ap2 for name in PPI]
            FEATURES_DIST[app + ap1 +ap2] = temp_PHISTS + temp_PPI 
    temp_PPI_all = [name + '_' + app + f'_{i}' for name in PPI for i in range(1, 3)]
    FEATURES_DIST[app + 'all'] = temp_PHISTS_all + temp_PPI_all

        


SCORE=[]

for dist in [
FEATURES_DIST['L111'],FEATURES_DIST['L112'],FEATURES_DIST['L121'],FEATURES_DIST['L122'],FEATURES_DIST['L131'],FEATURES_DIST['L132'],FEATURES_DIST['L141'],FEATURES_DIST['L142'],FEATURES_DIST['L151'],FEATURES_DIST['L152'],FEATURES_DIST['L161'],FEATURES_DIST['L162'], FEATURES_DIST['L1all'],
FEATURES_DIST['L211'],FEATURES_DIST['L212'],FEATURES_DIST['L221'],FEATURES_DIST['L222'],FEATURES_DIST['L231'],FEATURES_DIST['L232'],FEATURES_DIST['L241'],FEATURES_DIST['L242'],FEATURES_DIST['L251'],FEATURES_DIST['L252'],FEATURES_DIST['L261'],FEATURES_DIST['L262'], FEATURES_DIST['L2all'],
FEATURES_DIST['max11'],FEATURES_DIST['max12'],FEATURES_DIST['max21'],FEATURES_DIST['max22'],FEATURES_DIST['max31'],FEATURES_DIST['max32'],FEATURES_DIST['max41'],FEATURES_DIST['max42'],FEATURES_DIST['max51'],FEATURES_DIST['max52'],FEATURES_DIST['max61'],FEATURES_DIST['max62'], FEATURES_DIST['maxall'],
FEATURES_DIST['cos11'],FEATURES_DIST['cos12'],FEATURES_DIST['cos21'],FEATURES_DIST['cos22'],FEATURES_DIST['cos31'],FEATURES_DIST['cos32'],FEATURES_DIST['cos41'],FEATURES_DIST['cos42'], FEATURES_DIST['cos51'],FEATURES_DIST['cos52'],FEATURES_DIST['cos61'],FEATURES_DIST['cos62'], FEATURES_DIST['cosall'],
FEATURES_DIST['JS11'],FEATURES_DIST['JS12'],FEATURES_DIST['JS21'],FEATURES_DIST['JS22'],FEATURES_DIST['JS31'],FEATURES_DIST['JS32'],FEATURES_DIST['JS41'],FEATURES_DIST['JS42'],FEATURES_DIST['JS51'],FEATURES_DIST['JS52'],FEATURES_DIST['JS61'],FEATURES_DIST['JS62'], FEATURES_DIST['JSall'],
FEATURES_DIST['W11'],FEATURES_DIST['W12'],FEATURES_DIST['W21'],FEATURES_DIST['W22'],FEATURES_DIST['W31'],FEATURES_DIST['W32'],FEATURES_DIST['W41'],FEATURES_DIST['W42'],FEATURES_DIST['W51'],FEATURES_DIST['W52'],FEATURES_DIST['W61'],FEATURES_DIST['W62'],FEATURES_DIST['Wall'],
FEATURES_DIST['CM11'],FEATURES_DIST['CM12'],FEATURES_DIST['CM21'],FEATURES_DIST['CM22'],FEATURES_DIST['CM31'],FEATURES_DIST['CM32'],FEATURES_DIST['CM41'],FEATURES_DIST['CM42'],FEATURES_DIST['CM51'],FEATURES_DIST['CM52'],FEATURES_DIST['CM61'],FEATURES_DIST['CM62'],FEATURES_DIST['CMall']
]:  
   
    print('-'*50)
    Xtrain = df_train[dist + features_B + features_S]
    ytrain = df_train.SM_CATEGORY
    Xval = df_val[dist + features_B+ features_S]
    yval = df_val.SM_CATEGORY


    start = timer()
    knn = KNeighborsClassifier(n_neighbors = 5, weights = 'distance', p= 0.1)
    knn.fit(Xtrain, ytrain)
    predicted = knn.predict(Xval)
    end = timer()

    score = metrics.accuracy_score(yval, predicted)
    SCORE.append(score)
    
    print('validation accuracy:',score)
    print('execution time:', timedelta(seconds=end-start))
print(SCORE)


import pickle
with open('results/compare_dist_results.pickle', 'wb') as handle:
    pickle.dump(SCORE, handle, protocol=pickle.HIGHEST_PROTOCOL)