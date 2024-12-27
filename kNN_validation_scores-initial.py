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

#####################################################################################
# target
target_used = 'SM_CATEGORY' # numerical representation of Service
# features
features_list = {
                 'features_basic' : features_B,
                 'features_basic_statistical' : features_B + features_S
                }

# metric parameters
p_list = [0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.2, 0.3, 0.5, 0.7, 1]

# kNN parameters
n_neighbors_list = [1, 3, 5, 7, 9]
weights_list = ['uniform','distance']

#####################################################################################

print('Loading data...')
df_train = pd.read_pickle('data/preprocessed/metric_features_preprocessed_sample_10000_0.012_train.pkl')
df_val = pd.read_pickle('data/preprocessed/metric_features_preprocessed_sample_10000_0.012_val.pkl')
#df_test = pd.read_pickle('data/preprocessed/metric_features_preprocessed_sample_10000_0.012_test.pkl')

# list to keep the results
results = []

##############################################################################################
print('Starting evaluation...')

for feat_names in features_list.keys():
    print('-'*100)
    print('Features:', feat_names)
    features_used = features_list[feat_names]
    
    # data preparation
    Xtrain, ytrain, Xval, yval = prepare_XY(df_train, df_val, features_used, target_used)
    print('-'*50)
    
    for p in p_list:
        print('Metric p =', p)
                        
        for n_neighbors in n_neighbors_list:
            print('k =', n_neighbors)
            for weights in weights_list:
                
                # scores on validation set
                class_accuracy, f1_mac = knn_scores(Xtrain, ytrain, Xval, yval, n_neighbors, weights, p)
                results.append((target_used, feat_names, p, n_neighbors, weights, class_accuracy, f1_mac))
            
            
print('-'*100)
print('Processing results...')
kNN_results = pd.DataFrame(results,columns=['target', 'features', 'p', 'n_neigh', 'weights', 'class_accuracy', 'f1_mac'])

kNN_results.to_pickle('results/kNN_validation_scores-initial.pkl')
print('Evaluation done!')