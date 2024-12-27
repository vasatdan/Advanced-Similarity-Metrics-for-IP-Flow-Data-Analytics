
import pandas as pd
import numpy as np
import sklearn.metrics as metrics
from timeit import default_timer as timer
from datetime import timedelta
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import minkowski
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

import warnings
warnings.simplefilter('ignore', UserWarning)

start = timer()
print("Reading data")
# path to data prepared by script /vnat/03_metric_features_extraction.py
vnat_train = pd.read_pickle("data/VNAT_preprocessed/metric_features_preprocessed_vnat_dataset_train.pkl" )
vnat_test = pd.read_pickle("data/VNAT_preprocessed/metric_features_preprocessed_vnat_dataset_test.pkl" )

end = timer()
print('load time:', timedelta(seconds=end-start))


reduced =  ['BYTES', 'BYTES_REV', 'BYTES_TOTAL_RATE', 'FIN_COUNT', 'LENGTHS_STD',
       'FWD_LENGTHS_MIN', 'FWD_LENGTHS_MAX', 'FWD_LENGTHS_MEAN',
       'FWD_LENGTHS_STD', 'BWD_LENGTHS_MIN', 'BWD_LENGTHS_MAX',
       'BWD_LENGTHS_MEAN', 'BWD_LENGTHS_STD', 'PKT_IAT_MAX', 'BWD_PKT_IAT_MAX',
       'BWD_PKT_IAT_MEAN', 'D_PHISTS_IPT_DIST_CM_1',
       'D_PHISTS_SIZES_DIST_CM_2', 'S_PHISTS_SIZES_DIST_CM_3',
       'D_PHISTS_SIZES_DIST_CM_4', 'D_PHISTS_SIZES_DIST_CM_5',
       'S_PHISTS_SIZES_DIST_CM_5', 'D_PHISTS_SIZES_DIST_CM_6',
       'D_PPI_IPT_DIST_CM_1', 'D_PPI_LENGTHS_DIST_CM_1', 'D_PPI_IPT_DIST_CM_2',
       'S_PPI_LENGTHS_DIST_CM_2', 'D_PPI_LENGTHS_DIST_CM_2']

Xtrain = vnat_train[reduced]
ytrain = vnat_train.LABEL
Xtest = vnat_test[reduced]
ytest = vnat_test.LABEL
print(Xtrain.shape, Xtest.shape)


start = timer()
knn3 = KNeighborsClassifier(n_neighbors = 3, weights = 'distance', p=0.1)
knn3.fit(Xtrain, ytrain)
predicted3 = knn3.predict(Xtest)
end = timer()


score_3 = metrics.accuracy_score(ytest, predicted3)
micro_f1 = f1_score(ytest, predicted3, average='micro')
print(f"Micro F1 Score: {micro_f1}")
macro_f1 = f1_score(ytest, predicted3, average='macro')
print(f"Macro F1 Score: {macro_f1}")
print('knn_3')
print(metrics.classification_report(ytest, predicted3, digits = 4))

matrix = confusion_matrix(ytest, predicted3)

print(matrix)
print('execution time:', timedelta(seconds=end-start))

# import pickle
# with open('data/vnat_predicted.pickle', 'wb') as handle:
#     pickle.dump(predicted3, handle, protocol=pickle.HIGHEST_PROTOCOL)
