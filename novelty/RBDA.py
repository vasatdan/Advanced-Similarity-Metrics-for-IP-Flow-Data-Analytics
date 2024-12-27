import math
import numpy as np
from sklearn.neighbors import NearestNeighbors

def RBDA(Xtrain, Xval, p, k, traindelta = 2000, valdelta = 5000):
    # prepare output
    rqp = np.zeros((Xval.shape[0],))
    # prepare for split the trainig set
    trainsize = Xtrain.shape[0]
    # Nearest neighbors for p
    neigh = NearestNeighbors(n_neighbors = k, p = p)
    neigh.fit(Xtrain)
    # Get validation neighbors
    neigh_dist_val, neigh_ind_val = neigh.kneighbors(Xval, n_neighbors = k)
    # split the validation set
    valsize = Xval.shape[0]
    for vali in range(math.ceil(valsize/valdelta)):
        mini = vali*valdelta
        maxi = min(valsize,(vali+1)*valdelta)
        # temp val
        _Xval = Xval.iloc[mini:maxi,:]
        # Get local neighbors
        _neigh_dist_val, _neigh_ind_val = neigh_dist_val[mini:maxi,:], neigh_ind_val[mini:maxi,:]
        for tri in range(math.ceil(trainsize/traindelta)):
            trmini = tri*traindelta
            trmaxi = min(trainsize,(tri+1)*traindelta)
            # temp train
            _Xtrain = Xtrain.iloc[trmini:trmaxi,:]
            # Predicting train
            _neigh_dist_train, _neigh_ind_train = neigh.radius_neighbors(_Xtrain, radius = _neigh_dist_val.max(), return_distance=True)
            _rqp = np.zeros((_Xval.shape[0],k))
            for j in range(_Xval.shape[0]):
                for i in range(k):
                    if _neigh_ind_val[j,i] >= trmini and _neigh_ind_val[j,i] < trmaxi:
                        # subtract one for a point as self neighbor
                        _rqp[j,i] = np.sum(_neigh_dist_train[_neigh_ind_val[j,i] - trmini] < _neigh_dist_val[j,i]) - 1
            # collect
            rqp[mini:maxi] = rqp[mini:maxi] + _rqp.sum(axis = 1)
    return rqp/k