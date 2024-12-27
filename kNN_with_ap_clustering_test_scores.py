import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.metrics import pairwise_distances

from timeit import default_timer as timer
from datetime import timedelta

import warnings
warnings.simplefilter('ignore', UserWarning)

# set random seed for Affinity Propagation
rd_seed = 42

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
    Function trains kNN and calculates classification accuracy, macro average F1 score, and time length of trainig and prediction of kNN.
    """
    # train model
    start = timer()
    knn = KNeighborsClassifier(n_neighbors = n_neighbors, weights = weights, p=p).fit(Xtrain, ytrain)
    end = timer()
    time_t = timedelta(seconds=end-start)
    
    # predict
    start = timer()
    predicted = knn.predict(Xeval)
    end = timer()
    time_p = timedelta(seconds=end-start)
    
    # calculate scores
    class_accuracy = metrics.accuracy_score(yeval, predicted)
    f1_mac = metrics.f1_score(yeval, predicted, average='macro')
    
    return class_accuracy, f1_mac, time_t, time_p

def ap_clustering(Xtrain, p, damping, preference_quantile, max_iter):
    """
    Function finds clustering of training set using Affinity Propagation. Returns indices of cluster centers.
    """
    # calculate distances between points
    dist = pairwise_distances(Xtrain,metric='minkowski',p=p)
    # set initial similarities
    similarity = -((dist)**p)
    # set preference as quantile of similarities
    preference = np.quantile(similarity,preference_quantile)
    
    # run clustering
    ap = AffinityPropagation(damping=damping,
                             preference=preference,
                             affinity='precomputed',
                             max_iter=max_iter,
                             random_state=rd_seed).fit(similarity)
    
    center_idx = ap.cluster_centers_indices_
    n_clusters = center_idx.shape[0]    
    converge = int(ap.n_iter_ < max_iter)
    
    return center_idx, converge, n_clusters

def ap_center_dataset(df_train, features_used, target_used, p, damping, preference_quantile, max_iter):
    """
    Function finds Affinity Propagation clustering of each class in training set. Returns cluster centers.
    """
    df_train_reset = df_train.reset_index()
    
    # find unique values of target
    classes_list = df_train_reset[target_used].unique()
    
    centers_idx_list = []
    
    for target_class in classes_list:
        # select target category
        df_train_category = df_train_reset[df_train_reset[target_used] == target_class]
        Xtrain_category = df_train_category[features_used]
        
        # cluster category
        center_idx, converge, n_clusters = ap_clustering(Xtrain_category, p, damping, preference_quantile, max_iter)
        
        # add category centers to list of centers
        centers_idx_list += list(df_train_category.iloc[center_idx].index)
        
    return df_train_reset.iloc[centers_idx_list]

def prepare_centers_XY(df_train, features_used, target_used, p, damping, preference_quantile, max_iter):
    """
    Function selects features and target of training data cluster centers.
    """
    # find cluster centers
    df_train_centers = ap_center_dataset(df_train, features_used, target_used, p, damping, preference_quantile, max_iter)
    
    Xtrain_centers = df_train_centers[features_used]
    ytrain_centers = df_train_centers[target_used]
    
    return Xtrain_centers, ytrain_centers

def knn_scores_sample(Xtrain, ytrain, Xeval, yeval, n_neighbors, weights, p, n_centers, repetitions):
    """
    Function trains kNN on random samples of training set (of size n_centers) and calculates classification accuracy, macro average F1 score, and time length of trainig and prediction of kNN.
    """
    ca = []
    f1m = []
    tt = []
    tp = []
    
    # run for number of repetitions
    for rep in range(repetitions):
        # generate sample
        sample_idx = np.random.randint(Xtrain.shape[0],size = n_centers)        
        Xtrain_sample = Xtrain.iloc[sample_idx]
        ytrain_sample = ytrain.iloc[sample_idx]
        
        # evaluate kNN
        c_a, f1_mac, t_t, t_p = knn_scores(Xtrain_sample, ytrain_sample, Xeval, yeval, n_neighbors, weights, p)
        ca.append(c_a)
        f1m.append(f1_mac)
        tt.append(t_t)
        tp.append(t_p)
               
    class_accuracy = np.array(ca)
    f1_mac = np.array(f1m)
    time_t = np.array(tt)
    time_p = np.array(tp)
    
    # return values as arrays
    return class_accuracy, f1_mac, time_t, time_p

#####################################################################################
# feature sets
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
target_used = 'SM_CATEGORY' # numerical representation of Service
# features
features_names = 'features_reduced'
features_used = features_reduced

# metric parameters
p = 0.1

# kNN parameters
n_neighbors = 3
weights = 'distance'

# Affinity Propagation parameters
damping = 0.8
preference_quantile_list = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
max_iter = 200

# number of random samples
sample_reps = 5

#####################################################################################

print('Loading data...')
df_train = pd.read_pickle('data/preprocessed/metric_features_preprocessed_sample_10000_0.012_train.pkl')
#df_val = pd.read_pickle('data/preprocessed/metric_features_preprocessed_sample_10000_0.012_val.pkl')
df_test = pd.read_pickle('data/preprocessed/metric_features_preprocessed_sample_10000_0.012_test.pkl')

# list to keep the results
results = []

##############################################################################################
print('Starting evaluation...')

# data preparation
Xtrain, ytrain, Xtest, ytest = prepare_XY(df_train, df_test, features_used, target_used)
print('-'*100)

# scores on test set, using whole training set
print('Evaluating on full training set...')
class_accuracy, f1_mac, time_t, time_p = knn_scores(Xtrain, ytrain, Xtest, ytest, n_neighbors, weights, p)
print('-'*100)

# clustering for different values of preference quantile
for preference_quantile in preference_quantile_list:
    print('-'*50)
    print('Preference quantile: ', preference_quantile)
    # find cluster centers in training set
    print('Clustering training set...')
    Xtrain_centers, ytrain_centers = prepare_centers_XY(df_train, features_used, target_used, p, damping, preference_quantile, max_iter)
    
    # check if there are enough centers
    if Xtrain_centers.shape[0] >= n_neighbors:
        # scores on test set, using cluster centers as training set
        print('Evaluating with respect to cluster centers...')
        class_accuracy_c, f1_mac_c, time_t_c, time_p_c = knn_scores(Xtrain_centers, ytrain_centers, Xtest, ytest, n_neighbors, weights, p)
        # scores on test set, using randomly selected subsets as training set, random subsets have same size as cluster centers
        print('Evaluating with respect to random samples...')
        class_accuracy_s, f1_mac_s, time_t_s, time_p_s = knn_scores_sample(Xtrain, ytrain, Xtest, ytest, n_neighbors, weights, p, Xtrain_centers.shape[0], sample_reps)
        
    else:
        class_accuracy_c = 0
        f1_mac_c = 0
        time_t_c = timedelta()
        time_p_c = timedelta()
        
        f1_mac_s = [0]*sample_reps
        class_accuracy_s = [0]*sample_reps
        time_t_s = [timedelta()]*sample_reps
        time_p_s = [timedelta()]*sample_reps
        
    results.append((target_used, features_names, p, n_neighbors, weights, damping, preference_quantile, Xtrain_centers.shape[0], sample_reps, class_accuracy, class_accuracy_c, class_accuracy_s, f1_mac, f1_mac_c, f1_mac_s, time_t, time_t_c, time_t_s, time_p, time_p_c, time_p_s))
        
print('-'*100)
print('Processing results...')
kNN_results = pd.DataFrame(results,columns=['target', 'features', 'p', 'n_neigh', 'weights', 'damping', 'pref_q', 'n_clust', 'sample_reps', 'class_accuracy', 'class_accuracy_c', 'class_accuracy_s', 'f1_mac', 'f1_mac_c', 'f1_mac_s', 'time_t', 'time_t_c', 'time_t_s', 'time_p', 'time_p_c', 'time_p_s'])

kNN_results.to_pickle('results/kNN_with_ap_clustering_test_scores.pkl')
print('Evaluation done!')