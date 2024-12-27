
import pandas as pd
import numpy as np

from scipy.spatial.distance import minkowski
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as metrics
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.model_selection import train_test_split

import warnings
warnings.simplefilter('ignore', UserWarning)

print('loading data')

# path to data computed by script 05_metric_features_extraction.py
df_train = pd.read_pickle("data/preprocessed/metric_features_preprocessed_sample_200000_0.001_train.pkl")
df_val = pd.read_pickle("data/preprocessed/metric_features_preprocessed_sample_200000_0.001_val.pkl")


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
features_dist_CM_all = ['D_PHISTS_SIZES_DIST_CM_1', 'S_PHISTS_SIZES_DIST_CM_1', 'D_PHISTS_IPT_DIST_CM_1',
                        'S_PHISTS_IPT_DIST_CM_1','D_PHISTS_SIZES_DIST_CM_2', 'S_PHISTS_SIZES_DIST_CM_2', 'D_PHISTS_IPT_DIST_CM_2',
                        'S_PHISTS_IPT_DIST_CM_2','D_PHISTS_SIZES_DIST_CM_3', 'S_PHISTS_SIZES_DIST_CM_3', 'D_PHISTS_IPT_DIST_CM_3',
                        'S_PHISTS_IPT_DIST_CM_3','D_PHISTS_SIZES_DIST_CM_4', 'S_PHISTS_SIZES_DIST_CM_4', 'D_PHISTS_IPT_DIST_CM_4',
                        'S_PHISTS_IPT_DIST_CM_4','D_PHISTS_SIZES_DIST_CM_5', 'S_PHISTS_SIZES_DIST_CM_5', 'D_PHISTS_IPT_DIST_CM_5',
                        'S_PHISTS_IPT_DIST_CM_5','D_PHISTS_SIZES_DIST_CM_6', 'S_PHISTS_SIZES_DIST_CM_6', 'D_PHISTS_IPT_DIST_CM_6',
                        'S_PHISTS_IPT_DIST_CM_6', 'S_PPI_IPT_DIST_CM_1', 'D_PPI_IPT_DIST_CM_1', 'S_PPI_LENGTHS_DIST_CM_1',
                        'D_PPI_LENGTHS_DIST_CM_1','S_PPI_IPT_DIST_CM_2', 'D_PPI_IPT_DIST_CM_2', 'S_PPI_LENGTHS_DIST_CM_2',
                        'D_PPI_LENGTHS_DIST_CM_2']


features = features_B +features_S + features_dist_CM_all


Xtrain = df_train[features]  
ytrain = df_train.SM_CATEGORY
Xval = df_val[features] 
yval = df_val.SM_CATEGORY

n = Xtrain.shape[1]
print(n)
m = len(ytrain.unique())
print(m)

#pca
print('reduction and kNN')



knn_2 = KNeighborsClassifier(n_neighbors = 5, weights = 'distance')
knn_01 = KNeighborsClassifier(n_neighbors = 5, weights = 'distance', p=0.1)

print('Euclides')
knn_2.fit(Xtrain, ytrain)
predicted_2 = knn_2.predict(Xval)
score_n_2 = metrics.accuracy_score(yval, predicted_2)
print(score_n_2)

print('l_01')
knn_01.fit(Xtrain, ytrain)
predicted_01 = knn_01.predict(Xval)
score_n_01 = metrics.accuracy_score(yval, predicted_01)
print(score_n_01)

print('pca Euclides')
best_score_2 = 0

for number in range(1,n+1):
    
    pca = PCA(n_components= number)
    # fit and transform data
    Xtrain_pca = pca.fit_transform(Xtrain)
    Xval_pca = pca.transform(Xval)
    
    knn_2.fit(Xtrain_pca, ytrain)
    predicted_2 = knn_2.predict(Xval_pca)
    score_pca_n_2= metrics.accuracy_score(yval, predicted_2)
    if score_pca_n_2 > best_score_2:
        best_score_2 = score_pca_n_2
        print(best_score_2, number)

print('pca l_0.1')


best_score_01 = 0
for number in range(1,n+1):
   
    pca = PCA(n_components= number)
    # fit and transform data
    Xtrain_pca = pca.fit_transform(Xtrain)
    Xval_pca = pca.transform(Xval)
    
    knn_01.fit(Xtrain_pca, ytrain)
    predicted_01 = knn_01.predict(Xval_pca)
    score_pca_n_01= metrics.accuracy_score(yval, predicted_01)
       
    if score_pca_n_01 > best_score_01:
        best_score_01 = score_pca_n_01
        print(best_score_01, number)



print('lda Euclides')

best_score_2 = 0

for number in range(1,m):
    
    lda = LinearDiscriminantAnalysis(n_components = number)
    # fit and transform data
    Xtrain_lda = lda.fit_transform(Xtrain,ytrain)
    Xval_lda = lda.transform(Xval)
    
    knn_2.fit(Xtrain_lda, ytrain)
    predicted_2 = knn_2.predict(Xval_lda)
    score_lda_n_2= metrics.accuracy_score(yval, predicted_2)
    if score_lda_n_2 > best_score_2:
        best_score_2 = score_lda_n_2
        print(best_score_2, number)


print('lda l_0.1')


best_score_01 = 0
for number in range(1,m):
    
    lda = LinearDiscriminantAnalysis(n_components = number)
    # fit and transform data
    Xtrain_lda = lda.fit_transform(Xtrain,ytrain)
    Xval_lda = lda.transform(Xval)
    
    knn_01.fit(Xtrain_lda, ytrain)
    predicted_01 = knn_01.predict(Xval_lda)
    score_lda_n_01= metrics.accuracy_score(yval, predicted_01)
        
    if score_lda_n_01 > best_score_01:
        best_score_01 = score_lda_n_01
        print(best_score_01, number)


print('nca Euklides')

best_score_2 = 0


for number in range(1,n+1):
    nca = NeighborhoodComponentsAnalysis(n_components = number)
    # fit and transform data
    Xtrain_nca = nca.fit_transform(Xtrain,ytrain)
    Xval_nca = nca.transform(Xval)
    
    knn_2.fit(Xtrain_nca, ytrain)
    predicted_2 = knn_2.predict(Xval_nca)
    score_nca_n_2= metrics.accuracy_score(yval, predicted_2)
    if score_nca_n_2 > best_score_2:
        best_score_2 = score_nca_n_2
        print(best_score_2,number)
    


print('nca l_0.1')

best_score_01 = 0

for number in range(1,n+1):
    nca = NeighborhoodComponentsAnalysis(n_components = number)
    # fit and transform data
    Xtrain_nca = nca.fit_transform(Xtrain,ytrain)
    Xval_nca = nca.transform(Xval)
    
    knn_01.fit(Xtrain_nca, ytrain)
    predicted_01 = knn_01.predict(Xval_nca)
    score_nca_n_01= metrics.accuracy_score(yval, predicted_01)
        
    if score_nca_n_01 > best_score_01:
        best_score_01 = score_nca_n_01
        print(best_score_01, number)









