# libraries
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import optuna
from optuna.trial import TrialState
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter('ignore', UserWarning)

####################################################
from model_functions import *
from training_functions import *
####################################################
# prepare save directories
save_dir = os.path.join("./results")
model_save_dir = os.path.join("./saved_models")
Path(save_dir).mkdir(parents=True, exist_ok=True)
Path(model_save_dir).mkdir(parents=True, exist_ok=True)
####################################################
print('Loading data.')
df_train = pd.read_pickle('../data/preprocessed/neural_network_preprocessed_sample_10000_0.012_train.pkl')
df_val = pd.read_pickle('../data/preprocessed/neural_network_preprocessed_sample_10000_0.012_val.pkl')
df_test = pd.read_pickle('../data/preprocessed/neural_network_preprocessed_sample_10000_0.012_test.pkl')

# df_train = df_train.iloc[0:1000,:]
# df_val = df_val.iloc[0:1000,:]
# df_test = df_test.iloc[0:1000,:]
####################################################
print('Data loaded.')
print('Train shape: ', df_train.shape, ', val shape: ', df_val.shape, ', test shape: ', df_test.shape)
####################################################
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
####################################################
# set feature names
PHISTS = ['D_PHISTS_SIZES', 'S_PHISTS_SIZES', 'D_PHISTS_IPT', 'S_PHISTS_IPT']
PPI = ['PPI_PKT_FLAGS', 'PPI_PKT_TIMES', 'PPI_PKT_LENGTHS', 'PPI_PKT_DIRECTIONS', 'PPI_PKT_IPT']
####################################################
# Normalization
print("Normalizing data...")
# OTHER first
scaler = StandardScaler()
columns = df_train.select_dtypes(exclude="object").drop(columns=["SM_CATEGORY"]).columns
scaler.fit(df_train[columns].astype(np.float32).values)
for _df in [df_train, df_val, df_test]:
    _df[columns] = _df[columns].astype('float32')
    _df[columns] = scaler.transform(_df[columns].values)

for col in PHISTS + PPI:
    if col.endswith("DIRECTIONS"):
        # let it be
        pass
    elif col.endswith("FLAGS"):
        # divide by maximal number
        _max = np.stack(df_train[col]).max()
        for _df in [df_train, df_val, df_test]:
            _df[col] = _df[col]/_max
    else:
        # standardization
        _mean = np.stack(df_train[col]).mean()
        _std = np.stack(df_train[col]).std()
        for _df in [df_train, df_val, df_test]:
            _df[col] = (_df[col] - _mean)/_std     
####################################################
# CUDA
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("CUDA available:", torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
####################################################
# dataset creation
def get_custom_dataset(df):
    # target
    y_tensor = torch.tensor(df["SM_CATEGORY"].values)
    # other features
    x_other = df.select_dtypes(exclude="object").drop(columns=["SM_CATEGORY"]).values
    x_other = torch.tensor(x_other).float()
    # PHIST
    data = []
    for col in PHISTS:
        data.append(np.stack(df[col]))
    x_phist = np.stack(data,axis = 1).astype(np.float32)
    x_phist = torch.tensor(x_phist).float()
    # PPI
    data = []
    for col in PPI:
        data.append(np.stack(df[col]))
    x_ppi = np.stack(data,axis = 1).astype(np.float32)
    x_ppi = torch.tensor(x_ppi).float()
    # call the master
    return TensorDataset(x_ppi, x_phist, x_other, y_tensor)
####################################################
# create datasets and data loaders
c_train_data = get_custom_dataset(df_train) 
c_val_data = get_custom_dataset(df_val) 
c_test_data = get_custom_dataset(df_test) 

training_loader = DataLoader(c_train_data, batch_size=32, shuffle=True)
validation_loader = DataLoader(c_val_data, batch_size=32, shuffle=False)
test_loader = DataLoader(c_test_data, batch_size=32, shuffle=False)
####################################################
# Model initialization and sample run
dataiter = iter(training_loader)
x_ppi, x_ph, x_o, labels = next(dataiter)
x_ppi, x_ph, x_o, labels  = x_ppi.to(device),x_ph.to(device),x_o.to(device),labels.to(device)
print("Batch from training data shapes:")
print(x_ppi.shape, x_ph.shape, x_o.shape, labels.shape)
# model definition
model = MultimodalClassificationModel()
model.to(device)
output = model(x_ppi, x_ph, x_o)
print(f"Sample model output shape {output.shape}")
####################################################
# Optuna objective
def objective(trial):
    # hyperparameters range set up based on preliminary experiments
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    # we optimize the number of hidden units, number of sequential blocks, dropout ratio, etc.
    params = {
        "cnn_final_len": trial.suggest_categorical("cnn_final_len", [2,5,10]),
        "cnn_phist_final_len": trial.suggest_categorical("cnn_phist_final_len", [2,4]),
        "ppi_unit1": trial.suggest_int("ppi_unit1", 100, 300, step = 10),
        "ppi_unit2": trial.suggest_int("ppi_unit2", 200, 350, step = 10),
        "phist_unit1": trial.suggest_int("phist_unit1", 10, 100, step = 5),
        "phist_unit2": trial.suggest_int("phist_unit2", 50, 200, step = 10),
        "others_unit": trial.suggest_int("others_unit", 200, 400, step = 10),
        "common_unit": trial.suggest_int("common_unit", 400, 800, step = 10),
        "cnn_num_hidden": trial.suggest_int("cnn_num_hidden", 1, 6),
        "cnn_phist_num_hidden": trial.suggest_int("cnn_phist_num_hidden", 1, 6),
        "others_num_hidden": trial.suggest_int("others_num_hidden", 1, 6),
        "dropout_rate": trial.suggest_float("dropout_rate", 0, 0.4, step=0.05)
    }
    # prepare model
    model = MultimodalClassificationModel(**params).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    # train
    print("Trial number:", trial.number)
    max_epochs = 45
    early_stopping_epochs = 4
    best_vloss, best_vacc = train_and_evaluate(model, max_epochs, loss_fn, optimizer, training_loader, validation_loader, device, model_save_dir, early_stopping_epochs, trial)
    print(f"Validation loss = {best_vloss}, accuracy = {best_vacc}")
    return best_vacc
####################################################
# Optimization study
sampler = optuna.samplers.TPESampler(seed = 42)
pruner = optuna.pruners.HyperbandPruner()
study = optuna.create_study(direction="maximize", sampler = sampler,pruner = pruner)
print(f"Running the optimization ...")
study.optimize(objective, n_trials = 60)
####################################################
print("\nNumber of finished trials: {}".format(len(study.trials)))
print("Best trial:")
trial = study.best_trial
print("  Number:",trial.number)
print("  Value:",trial.value)
# save name
save_name = os.path.join(save_dir,f"optuna_best_model_params.pkl")
# get best params
params = trial.params
learning_rate = params.pop("learning_rate")
print("Best params: ", params)
print("Best learning rate", learning_rate)
# Save the output
with open(save_name, 'wb') as file:
    pickle.dump((params,learning_rate), file)
####################################################
# train final model
print("\n#####################################")
print("Training the final model")
# inicialize
model = MultimodalClassificationModel(**params).to(device)
loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.3)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# train
max_epochs = 80
early_stopping_epochs = 8
best_vloss, best_vacc = train_and_evaluate(model, max_epochs, loss_fn, optimizer, training_loader, validation_loader, device, model_save_dir, early_stopping_epochs, None)
# SAVE
model_save_name = os.path.join(model_save_dir, f"best_model.pt")
print(f"\nSaving the final model")
torch.save(model.state_dict(), model_save_name)
####################################################
print(f"Evaluating the final model")
vloss, vacc, vypred = evaluate_model(model, loss_fn, validation_loader, device, return_predictions = True)
yval = df_val["SM_CATEGORY"].values
val_accuracy = metrics.accuracy_score(yval, vypred)
val_f1 = metrics.f1_score(yval, vypred, average='macro')
print(f'Validation accuracy = {val_accuracy:.4f}, F1 = {val_f1:.3f}')
tloss, tacc, typred = evaluate_model(model, loss_fn, test_loader, device, return_predictions = True)
ytest = df_test["SM_CATEGORY"].values
test_accuracy = metrics.accuracy_score(ytest, typred)
test_f1 = metrics.f1_score(ytest, typred, average='macro')
print(f'Test accuracy = {test_accuracy:.4f}, F1 = {test_f1:.4f}')

# Save the final output
print(f"\nSaving the final output")
with open(save_name, 'wb') as file:
    pickle.dump((params,learning_rate,val_accuracy, test_accuracy, val_f1, test_f1, yval, vypred, ytest, typred), file)