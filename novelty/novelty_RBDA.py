import pickle
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn import metrics
from pathlib import Path
from RBDA import RBDA
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
##########################################################################
# FEATURE SETS
features_B =  ['TIME_DIFF', 'BYTES', 'BYTES_REV', 'PACKETS', 'PACKETS_REV']
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

features_reduced = ['BYTES', 'BYTES_REV', 'BYTES_TOTAL_RATE', 'FIN_COUNT', 'LENGTHS_STD',
  'FWD_LENGTHS_MIN', 'FWD_LENGTHS_MAX', 'FWD_LENGTHS_MEAN',
  'FWD_LENGTHS_STD', 'BWD_LENGTHS_MIN', 'BWD_LENGTHS_MAX',
  'BWD_LENGTHS_MEAN', 'BWD_LENGTHS_STD', 'PKT_IAT_MAX', 'BWD_PKT_IAT_MAX',
  'BWD_PKT_IAT_MEAN', 'D_PHISTS_IPT_DIST_CM_1',
  'D_PHISTS_SIZES_DIST_CM_2', 'S_PHISTS_SIZES_DIST_CM_3',
  'D_PHISTS_SIZES_DIST_CM_4', 'D_PHISTS_SIZES_DIST_CM_5',
  'S_PHISTS_SIZES_DIST_CM_5', 'D_PHISTS_SIZES_DIST_CM_6',
  'D_PPI_IPT_DIST_CM_1', 'D_PPI_LENGTHS_DIST_CM_1', 'D_PPI_IPT_DIST_CM_2',
  'S_PPI_LENGTHS_DIST_CM_2', 'D_PPI_LENGTHS_DIST_CM_2']

feature_sets = [
  ("B+S",features_B+features_S),
  ("B+S+CM", features_B+features_S+features_CM_best),
  ("B+S+CM_all", features_B+features_S+features_CM_all),
  ("reduced", features_reduced)
]

##########################################################################
# prepare save directories
save_dir = os.path.join("../results/novelty")
Path(save_dir).mkdir(parents=True, exist_ok=True)

##########################################################################
# Common parameters
k = 3
ps = [0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.3, 0.5, 0.75, 1]

##########################################################################
print(f"\nIterating novelty fraction")
for novelty_frac in [0.25, 0.5]:
  print("\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
  print("-----------------------------------------------------")
  print(f"-------- Running novelty fraction = {novelty_frac}")
  #------------------------------------------
  print("  Loading data")
  general_path = "../data/preprocessed/"
  base_name = f"metric_features_novelty_{novelty_frac}_preprocessed_sample_10000_0.012"
  df_train = pd.read_pickle(os.path.join(general_path,f"{base_name}_train.pkl"))
  df_val = pd.read_pickle(os.path.join(general_path,f"{base_name}_val.pkl"))
  df_test = pd.read_pickle(os.path.join(general_path,f"{base_name}_test.pkl"))
  # find known categories
  known_categories = sorted(df_train.SM_CATEGORY.unique())
  print(f"Known categories ({len(known_categories)}):", known_categories)
  # indicate novel categories
  is_novel_test = (~df_test.SM_CATEGORY.isin(known_categories)).astype(int)
  #------------------------------------------
  print(f"  Iterating the feature sets")
  for i, feature_set_tuple  in enumerate(feature_sets):
    print("...................................................")
    feature_set_name = feature_set_tuple[0]
    print(f"........ Running i = {i}, {feature_set_name}")
    # get sets
    Xtrain = df_train[feature_set_tuple[1]]
    ytrain = df_train.SM_CATEGORY
    Xval = df_val[feature_set_tuple[1]]
    yval = df_val.SM_CATEGORY
    Xtest = df_test[feature_set_tuple[1]]
    ytest = df_test.SM_CATEGORY
    # --------------------
    # Calculation
    testrocaucs = []
    vallofs = []
    testlofs = []
    for p in ps:
      print(f"\tp = {p}")
      # Validation
      vlof = RBDA(Xtrain, Xval, p, k, traindelta = 5000, valdelta = 5000)
      vallofs.append(vlof)
      # Test
      tlof = RBDA(Xtrain, Xtest, p, k, traindelta = 5000, valdelta = 5000)
      testlofs.append(tlof)
      testrocaucs.append(metrics.roc_auc_score(is_novel_test, tlof, max_fpr = 0.1))
      print(f"\t - RBDA Test AUC = {testrocaucs[-1]}")
    # --------------------
    # save name
    save_name = os.path.join(save_dir,f"RBDA_{feature_set_name}_{base_name}.pkl")
    # save result every iteration
    print(f"\tSaving the output for {feature_set_name}")
    with open(save_name, 'wb') as file:
        pickle.dump((ps, novelty_frac, k, vallofs, testrocaucs, testlofs), file)

