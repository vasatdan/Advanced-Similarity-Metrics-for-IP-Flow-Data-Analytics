# Advanced-Similarity-Metrics-for-IP-Flow-Data-Analytics

This repository contain all source codes for the paper Advanced Similarity Metrics for IP Flow Data Analytics.

All codes are written in Python 3. The library requirements are in `python_requirements.txt` and can be installed by `pip install -r python_requirements.txt`.

### Dataset
For running the experiments, it is expected that the dataset https://zenodo.org/records/13808423 is located in `data/raw` (i.e., that there exists folder `data/raw/20220819`).

## Data preprocessing
Scripts responsible for preprocessing the raw data into a form suitable for experiments are located in the folder `data_preprocessing`.

The following data preprocessing steps need to be run first:
- `01_assign_sm_category.py` - loads raw data and performs ground truth labeling using the service map;
- `02_cleaning.py` - cleans the dataset;
- `03_sample.py` - makes a sample;
- `04_preprocessing.py` - changes types and prepares basic flow and statistical features;
- `05_split_and_prob_dist_features_extraction.py` - performs train/val/test split and calculates probability distance features;
- `06_novelty_split_and_features_extract.py` - performs train/val/test split, then selects 'novel' categories and removes them from val and test. Then it calculates probability distance features;
- `07_neural_network_split_and_features_extract.py` - produces dataset for neural network by  performing train/val/test split and dropping unnecessary features.

The output of those scripts are train/val/test data located in the folder `data/preprocessed`. The folder should contain 3 files `metric_features_preprocessed_...` for most of the experiments, 3 files `neural_network_...` for neural networks, and 6 files `metric_features_novelty_...` for 2 novelty scenarios.

## Initial choice of distance and kNN parameters
The script `kNN_validation_scores-initial.py` computes classification scores on the validation set for various kNN parameters, $l_p$-metrics, and basic flow and statistical features.

The output DataFrame is stored in folder `results/kNN_validation_scores-initial.pkl`.

The results are reported in the first part of `kNN_scores_visualization.ipynb` jupyter notebook.

## Selecting the distance and reference histogram and splt
The script `compare_probability_distance.py` computes classification accuracy for B+S+distance features (for all investigated combinations of distances and reference histograms and splt).

The results are saved to `results/compare_dist_results.pickle` and reported in `compare_distance_vizualization.ipynb`.

## Feature reduction
The script `feature_reduction.py` performs reduction of features for five best performing features vectors.
Reduced feature vectors are saved to `results/reduced_features_{random_seed}.pickle`.

The outputs are reported in `features_reduction_results.ipynb`.
                        
## Linear projection dimensionality reduction methods
For this part we utilize smaller dataset drawn from the dataset using the script `data_preprocessing/03_sample.py` with parameters setting `min_cases_in_class = 200 000, fraction_of_class = 0.001`  (followed by `04_preprocessing.py` and `05_split_and_prob_dist_features_extraction.py`).

In the `PcaLdaNca.py` script, for each method (PCA, LDA, NCA), every possible value of the number of components in a reasonable range is evaluated by the kNN classification accuracy on the validation set.
                        
## Final choice of kNN parameters
The script `kNN_validation_scores_B_S_CM_reduced.py` computes classification scores on validation set for various kNN parameters, $l_p$-metrics, and feature sets containing probability distance features.

The output DataFrame is stored in `results/kNN_validation_scores_BSCM_reduced.pkl` and reported in `kNN_scores_visualization.ipynb`.

## Calculation of kNN test scores
In `kNN_test_scores.py`, classification scores on test set for chosen kNN parameters and $l_p$-metric are computed.

The output DataFrame is stored in `results/kNN_test_scores.pkl` and reported in `kNN_scores_visualization.ipynb`.

## Lightgbm classification
The LightGBM classification is performed in `lightgbm_classification.py` script. It uses Optuna for hyperparameters tuning for each feature set. The final models are saved in folder `saved_models/lightgbm` and the results in `results/lightgbm` folder.

The report with the best hyperparameters is presented in `lightgbm_results.ipynb` notebook.

## Deep learning model for classification
All the scripts for deep learning model are located in the `neural_network` folder:
- `model_functions.py` - the definition of the model;
- `training_functions.py` - functions needed for the training and evaluation;
- `train_model.py` - main script for data loading and preprocessing followed by hyperparameter tuning and training of the final model. The best model is saved in `neural_network/saved_models/best_model.pt` and its hyperparameters in `neural_network/results/optuna_best_model_params.pkl`. 

The results are reported in `neural_network/show_results.ipynb`.

## Performance on VNAT dataset
For running this evaluation it is expected that the dataset is located in `data/VNAT_raw`.
Scripts responsible for preprocessing the raw VNAT data into a form suitable for evaluation are located in the folder `data_preprocessing/prepare_vnat_dataset`. 

The following data preprocessing steps need to be run first:
- `01_load_vnat_dataset.py` - loads raw data;
- `02_vnat_preprocessing.py` - cleans and preprocessed the dataset;
- `03_split_and_metric_features_extraction.py` - performs train/val/test split and prepares probability distance features.

### Evaluation
The script `knn_vnat.py` computes classification scores on test set of the VNAT dataset.

## Novelty detection
All the scripts for novelty detection are presented in the `novelty` folder:
- `novelty_LOF.py` - performs the evaluation of the Local Outlier Factor method;
- `RBDA.py` - the custom implementation of the Rank-Based Detection Algorithm;
- `novelty_RBDA.py` - perform the evaluation of the RBDA method;
- `novelty_lightgbm_train.py` - performs the hyperparameter optimization and training of the lightgbm models (individual for each novelty scenario and each feature set);
- `novelty_lightgbm_eval.py` - evaluation of the best lightgbm models trained by the previous script.

All results are stored in `results/novelty` and lightgbm models are saved in `saved_models/novelty/lightgbm`.

Obtained results are reported in `novelty/novelty_results.ipynb`.

## Accelerating kNN with AP clustering
The script `kNN_with_ap_clustering_test_scores.py` computes classification scores on test set for chosen kNN parameters and $l_p$-metric - kNN is performed with respect to full training set, set consisting of AP cluster centers and randomly drawn samples.

The output DataFrame is stored in `results/kNN_with_ap_clustering_test_scores.pkl` and reported in `kNN_scores_visualization.ipynb`.
                        
## Visualizations and plots
Jupyter notebooks generating visualizations and plots are:
- `voronoi_visualization.ipynb` - generates Voronoi diagrams for the $l_p$-metric;
- `compare_distances_vizualization.ipynb` - plots results of comparison of different probability distance measures;
- `kNN_scores_visualization.ipynb` - displays results of kNN classification accuracy with respect to different metrics, features and parameters of kNN algorithm, includes validation/test scores and scores obtained with clustering.