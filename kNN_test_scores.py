import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

import warnings
warnings.simplefilter('ignore', UserWarning)

#####################################################################################
# functions
def prepare_XY(df_train, df_eval, features_used, target_used):
    """
    Function selects features and target from training data and data on which we evaluate model.
    """
    Xtrain = df_train[features_used]
    Xeval = df_eval[features_used]

    ytrain = df_train[target_used]
    yeval = df_eval[target_used]
    
    return Xtrain, ytrain, Xeval, yeval

def knn_scores(Xtrain, ytrain, Xeval, yeval, n_neighbors, weights, p):
    """
    Function calculates classification accuracy and macro average F1 score of kNN.
    """
    # train model
    knn = KNeighborsClassifier(n_neighbors = n_neighbors, weights = weights, p=p).fit(Xtrain, ytrain)
    
    # predict
    predicted = knn.predict(Xeval)
    
    # calculate scores
    class_accuracy = metrics.accuracy_score(yeval, predicted)
    f1_mac = metrics.f1_score(yeval, predicted, average='macro')
    
    return class_accuracy, f1_mac
            
#####################################################################################
# feature sets
features_B = ['TIME_DIFF', 'BYTES', 'BYTES_REV', 'PACKETS', 'PACKETS_REV']

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

features_CM_all = ['D_PHISTS_SIZES_DIST_CM_1', 'S_PHISTS_SIZES_DIST_CM_1', 'D_PHISTS_IPT_DIST_CM_1',
                        'S_PHISTS_IPT_DIST_CM_1','D_PHISTS_SIZES_DIST_CM_2', 'S_PHISTS_SIZES_DIST_CM_2', 'D_PHISTS_IPT_DIST_CM_2',
                        'S_PHISTS_IPT_DIST_CM_2','D_PHISTS_SIZES_DIST_CM_3', 'S_PHISTS_SIZES_DIST_CM_3', 'D_PHISTS_IPT_DIST_CM_3',
                        'S_PHISTS_IPT_DIST_CM_3','D_PHISTS_SIZES_DIST_CM_4', 'S_PHISTS_SIZES_DIST_CM_4', 'D_PHISTS_IPT_DIST_CM_4',
                        'S_PHISTS_IPT_DIST_CM_4','D_PHISTS_SIZES_DIST_CM_5', 'S_PHISTS_SIZES_DIST_CM_5', 'D_PHISTS_IPT_DIST_CM_5',
                        'S_PHISTS_IPT_DIST_CM_5','D_PHISTS_SIZES_DIST_CM_6', 'S_PHISTS_SIZES_DIST_CM_6', 'D_PHISTS_IPT_DIST_CM_6',
                        'S_PHISTS_IPT_DIST_CM_6', 'S_PPI_IPT_DIST_CM_1', 'D_PPI_IPT_DIST_CM_1', 'S_PPI_LENGTHS_DIST_CM_1',
                        'D_PPI_LENGTHS_DIST_CM_1','S_PPI_IPT_DIST_CM_2', 'D_PPI_IPT_DIST_CM_2', 'S_PPI_LENGTHS_DIST_CM_2',
                        'D_PPI_LENGTHS_DIST_CM_2']

features_reduced =  ['BYTES', 'BYTES_REV', 'BYTES_TOTAL_RATE', 'FIN_COUNT', 'LENGTHS_STD',
       'FWD_LENGTHS_MIN', 'FWD_LENGTHS_MAX', 'FWD_LENGTHS_MEAN',
       'FWD_LENGTHS_STD', 'BWD_LENGTHS_MIN', 'BWD_LENGTHS_MAX',
       'BWD_LENGTHS_MEAN', 'BWD_LENGTHS_STD', 'PKT_IAT_MAX', 'BWD_PKT_IAT_MAX',
       'BWD_PKT_IAT_MEAN', 'D_PHISTS_IPT_DIST_CM_1',
       'D_PHISTS_SIZES_DIST_CM_2', 'S_PHISTS_SIZES_DIST_CM_3',
       'D_PHISTS_SIZES_DIST_CM_4', 'D_PHISTS_SIZES_DIST_CM_5',
       'S_PHISTS_SIZES_DIST_CM_5', 'D_PHISTS_SIZES_DIST_CM_6',
       'D_PPI_IPT_DIST_CM_1', 'D_PPI_LENGTHS_DIST_CM_1', 'D_PPI_IPT_DIST_CM_2',
       'S_PPI_LENGTHS_DIST_CM_2', 'D_PPI_LENGTHS_DIST_CM_2']

#####################################################################################
# target
targets_list = ['DOMAINS_COLLECTED',
                'SM_CATEGORY',
                'SERVICE CATEGORY']
# features
features_list = {
                 'features_basic_statistical' : features_B + features_S,
                 'features_basic_statistical_CM_best' : features_B + features_S + features_CM_best,
                 'features_basic_statistical_CM_all' : features_B + features_S + features_CM_all,
                 'features_reduced' : features_reduced
                }

# metric parameters
p = 0.1

# kNN parameters
n_neighbors = 3
weights = 'distance'

#####################################################################################

print('Loading data...')
df_train = pd.read_pickle('data/preprocessed/metric_features_preprocessed_sample_10000_0.012_train.pkl')
#df_val = pd.read_pickle('data/preprocessed/metric_features_preprocessed_sample_10000_0.012_val.pkl')
df_test = pd.read_pickle('data/preprocessed/metric_features_preprocessed_sample_10000_0.012_test.pkl')

# list to keep the results
results = []

##############################################################################################
print('Starting evaluation...')

for target_used in targets_list:
    print('-'*100)
    print('Target: ', target_used)
    
    for feat_names in features_list.keys():
        print('-'*50)
        print('Features: ', feat_names)
        features_used = features_list[feat_names]
        
        # data preparation
        Xtrain, ytrain, Xtest, ytest = prepare_XY(df_train, df_test, features_used, target_used)
        
        # scores on test set
        class_accuracy, f1_mac = knn_scores(Xtrain, ytrain, Xtest, ytest, n_neighbors, weights, p)
        results.append((target_used, feat_names, p, n_neighbors, weights, class_accuracy, f1_mac))
        
        
print('-'*100)
print('Processing results...')
kNN_results = pd.DataFrame(results,columns=['target', 'features', 'p', 'n_neigh', 'weights', 'class_accuracy', 'f1_mac'])

kNN_results.to_pickle('results/kNN_test_scores.pkl')
print('Evaluation done!')