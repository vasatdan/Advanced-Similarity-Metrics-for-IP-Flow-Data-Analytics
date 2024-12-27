import pickle
import os
import pandas as pd
import numpy as np
from sklearn import metrics
from pathlib import Path
import lightgbm as lgb
import optuna

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
model_save_dir = os.path.join("../saved_models/novelty/lightgbm")
Path(model_save_dir).mkdir(parents=True, exist_ok=True)

##########################################################################
print(f"\nIterating novelty fraction")
for novelty_frac in [0.25, 0.5]:
  print("\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
  print("-----------------------------------------------------")
  print(f"-------- Running novelty fraction = {novelty_frac}")
  #####################################################################
  print("  Loading data")
  general_path = "../data/preprocessed/"
  base_name = f"metric_features_novelty_{novelty_frac}_preprocessed_sample_10000_0.012"
  df_train = pd.read_pickle(os.path.join(general_path,f"{base_name}_train.pkl"))
  df_val = pd.read_pickle(os.path.join(general_path,f"{base_name}_val.pkl"))
  df_test = pd.read_pickle(os.path.join(general_path,f"{base_name}_test.pkl"))
  print(f"Loaded data shapes: train - {df_train.shape}, validation - {df_val.shape}, test - {df_test.shape}.")
  # relabel to indices from 0 to #categories - 1
  orig_labels = sorted(list(df_train.SM_CATEGORY.unique()))
  labels_map = {orig_labels[i]:i for i in range(len(orig_labels))}
  print(f"\nLabels map for relabeling before novel: {labels_map}.")
  novel_labels = sorted(list(set(list(df_test.SM_CATEGORY.unique())) - set(orig_labels)))
  labels_map.update({novel_labels[i]:(i+len(orig_labels)) for i in range(len(novel_labels))})
  print(f"\nLabels map for relabeling: {labels_map}.")
  # relabel
  for data in [df_train, df_val, df_test]:
    data.SM_CATEGORY = data.SM_CATEGORY.map(labels_map)
  # get number of categories
  num_categories = df_train.SM_CATEGORY.nunique()
  test_num_categories = df_test.SM_CATEGORY.nunique()
  print(f"There are {num_categories} known categories in train.")
  print(f"There are {test_num_categories} known categories in test.")
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
    # LOAD TRAINED model
    # save name
    save_name = os.path.join(save_dir,f"lightgbm_{feature_set_name}_{base_name}.pkl")
    old_save_name = os.path.join(save_dir,f"lightgbm_opt_params_{feature_set_name}_{base_name}.pkl")
    model_save_name = os.path.join(model_save_dir, f"model_{feature_set_name}_{base_name}.txt")
    # load model
    print("\tLoading the final model")
    model = lgb.Booster(model_file=model_save_name)
    # Load the output
    print("\tLoading the output of training")
    with open(old_save_name, 'rb') as file:
      trained_feature_set_tuple, study, params = pickle.load(file)
    print("\t  Params: ", params)
    # --------------------
    # PREDICT
    yvproba = model.predict(Xval, num_iteration=model.best_iteration)
    yvrawproba = model.predict(Xval, num_iteration=model.best_iteration, raw_score = True)
    yvpred = np.argmax(yvproba, axis=1)
    val_accuracy = metrics.accuracy_score(yval, yvpred)
    print(f'Validation accuracy: {val_accuracy}')
    ytproba = model.predict(Xtest, num_iteration=model.best_iteration)
    ytrawproba = model.predict(Xtest, num_iteration=model.best_iteration, raw_score = True)
    print(f'Test shape: {ytproba.shape}')
    # indicators of novelty for test
    novel_test = ytest >= len(orig_labels)
    # Save the final output
    with open(save_name, 'wb') as file:
      pickle.dump((feature_set_tuple, params, yvproba, yvrawproba, val_accuracy, ytproba, ytrawproba, ytest, novel_test, labels_map, orig_labels, novel_labels), file)


