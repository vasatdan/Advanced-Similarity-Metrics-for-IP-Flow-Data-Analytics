import os
import pickle
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn import metrics
from pathlib import Path

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
save_dir = os.path.join("./results/lightgbm")
model_save_dir = os.path.join("./saved_models/lightgbm")
Path(save_dir).mkdir(parents=True, exist_ok=True)
Path(model_save_dir).mkdir(parents=True, exist_ok=True)
##########################################################################
print("Loading data")
general_path = "./data/preprocessed/"
base_name = "metric_features_preprocessed_sample_10000_0.012"
df_train = pd.read_pickle(os.path.join(general_path,f"{base_name}_train.pkl"))
df_val = pd.read_pickle(os.path.join(general_path,f"{base_name}_val.pkl"))
df_test = pd.read_pickle(os.path.join(general_path,f"{base_name}_test.pkl"))

print(f"Loaded data shapes: train - {df_train.shape}, validation - {df_val.shape}, test - {df_test.shape}.")

# relabel to indices from 0 to #categories - 1
orig_labels = sorted(list(df_train.SM_CATEGORY.unique()))
labels_map = {orig_labels[i]:i for i in range(len(orig_labels))}
print(f"\nLabels map for relabeling: {labels_map}.")
# relabel
for data in [df_train, df_val, df_test]:
  data.SM_CATEGORY = data.SM_CATEGORY.map(labels_map)
  if data.SM_CATEGORY.max() > len(orig_labels) -1:
    raise ValueError(f"There was a problem in relabeling the target.")
# get number of categories
num_categories = df_train.SM_CATEGORY.nunique()
print(f"There are {num_categories} categories to predict.")

##########################################################################
print(f"\nIterating the feature sets")
for i, feature_set_tuple  in enumerate(feature_sets):
  print("-----------------------------------------------------")
  print("-----------------------------------------------------")
  feature_set_name = feature_set_tuple[0]
  print(f"-------- Running i = {i}, {feature_set_name}")
  # get sets
  Xtrain = df_train[feature_set_tuple[1]]
  ytrain = df_train.SM_CATEGORY
  Xval = df_val[feature_set_tuple[1]]
  yval = df_val.SM_CATEGORY
  Xtest = df_test[feature_set_tuple[1]]
  ytest = df_test.SM_CATEGORY
  # Static parameters
  const_params = {
    "boosting_type": "gbdt",
    "objective": "multiclass",
    "metric": "multi_error",
    "num_class": num_categories,
    'force_col_wise': True,
    'feature_pre_filter': False,
    "feature_fraction": 0.8,
    "bagging_fraction": 1.0,
    "bagging_freq": 0,
    'learning_rate': 0.1,
    "min_data_in_leaf": 20,
    "seed": 6886,
  }
  # define the objective function
  def objective(trial):
    # LightGBM parameters
    params = const_params.copy()
    params.update({
      "max_depth": trial.suggest_int("max_depth", 8, 11),
      "num_leaves": trial.suggest_int("num_leaves", 330, 350),
      "lambda_l1": trial.suggest_float("lambda_l1", 0.5e-8, 5e-8),
      "lambda_l2": trial.suggest_float("lambda_l2", 1.0, 8.0),
    })
    # num rounds
    num_round = 350
    train_data = lgb.Dataset(Xtrain, label=ytrain, params={"verbosity": -1})
    val_data = lgb.Dataset(Xval, label=yval, reference=train_data, params={"verbosity": -1})
    # train
    model = lgb.train(params, train_data, num_round, valid_sets=[val_data], callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(50)])
    # predict
    yproba = model.predict(Xval, num_iteration=model.best_iteration)
    ypred = np.argmax(yproba, axis=1)
    # return accuracy
    return metrics.accuracy_score(yval, ypred)
  # Optimization study
  sampler = optuna.samplers.TPESampler(seed = 42)
  study = optuna.create_study(direction = "maximize", sampler = sampler)
  print(f"Running the optimization ...")
  study.optimize(objective, n_trials = 50)

  print("\nNumber of finished trials: {}".format(len(study.trials)))
  print("Best trial:")
  trial = study.best_trial
  print("  Value: {}".format(trial.value))
  print("  Params: ", trial.params)
  # save name
  save_name = os.path.join(save_dir,f"optuna_{feature_set_name}_{base_name}.pkl")
  model_save_name = os.path.join(model_save_dir, f"model_{feature_set_name}_{base_name}.txt")
  # get best params
  params = const_params.copy()
  params.update(trial.params)
  # Save the output
  with open(save_name, 'wb') as file:
    pickle.dump((feature_set_tuple, study, params), file)
  # train final model
  print("\n#####################################")
  print("Training the final model")
  # num rounds
  num_round = 450
  # prepare data
  train_data = lgb.Dataset(Xtrain, label=ytrain)
  val_data = lgb.Dataset(Xval, label=yval, reference=train_data)
  # train
  model = lgb.train(params, train_data, num_round, valid_sets=[val_data], callbacks=[lgb.early_stopping(stopping_rounds=15), lgb.log_evaluation(50)])
  # SAVE
  print(f"\nSaving the final model")
  model.save_model(model_save_name)
  print(f"Evaluating the final model")
  # predict
  vyproba = model.predict(Xval, num_iteration=model.best_iteration)
  vypred = np.argmax(vyproba, axis=1)
  val_accuracy = metrics.accuracy_score(yval, vypred)
  val_f1 = metrics.f1_score(yval, vypred, average='macro')
  print(f'Validation accuracy = {val_accuracy}, F1 = {val_f1}')
  typroba = model.predict(Xtest, num_iteration=model.best_iteration)
  typred = np.argmax(typroba, axis=1)
  accuracy = metrics.accuracy_score(ytest, typred)
  f1 = metrics.f1_score(ytest, typred, average='macro')
  print(f'Test accuracy = {accuracy}, F1 = {f1}')

  # Save the final output
  print(f"\nSaving the final output")
  with open(save_name, 'wb') as file:
    pickle.dump((feature_set_tuple, study, params, val_accuracy, accuracy, val_f1, f1, yval, vypred, ytest, typred), file)
