###
# IMPORTS
###

import os, random, torch
import numpy as np
import matplotlib.pyplot as plt
from pipeline.process_data import process_data
from pipeline.train_model import (process_RNN_data, define_model, train_model,
                                  define_long_model, train_long_model)
from pipeline.evaluate_model import (test_model, test_long_model, generate_predictions,
                                     logistic_regression_embeddings, kmplots, MaskedGRU_feature_importance)

##
# DEFINE CONSTANTS
###

# hyperparameters
max_norm = 0.5
l1_lambda = 0.0001
hidden_size = 128
batch_size = 50
num_epochs = 500
lr = 5e-05
test_trainval_ratio = 0.2
train_val_ratio = 0.2
num_layers = 3
patience = 40
early_stopping = True

# RNN hyperparameters
patience_long = 65
dropout = 0.7
lr_long = 0.001

# program parameters
cohort = "pMCIiAD"            # pMCIiAD pHCiAD
model_choice = "MaskedGRU"   # GRU, simpleRNN, MaskedGRU
eval = True
imputed = True
output_size = 1
seed = 134

random.seed = seed
np.random.seed = seed
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)

###
# RUN MODEL
###

print("--- DATA PROCESSING ---")
print("> Processing raw files...")
X, y, is_missing, time_missing, static_covariates, longitudinal_covariates, rids = process_data(cohort, imputed)
print("------ COMPLETED ------\n")

print("----- TRAIN MODELS ----")
print("> Splitting data into train, validation, and test sets...")
train_loader, val_loader, test_loader = process_RNN_data(imputed, X, y, is_missing, time_missing,
                                                         test_trainval_ratio, train_val_ratio, batch_size, seed)
print("> Defining BARS model...")
model, criterion, optimizer = define_model(model_choice, X.shape[2], X.shape[2], num_layers, dropout, lr)
print("> Training BARS model...")
model = train_model(model, criterion, optimizer, num_epochs, train_loader, val_loader,
                max_norm, early_stopping, patience, imputed, model_choice)
print("> Testing BARS model performance metrics...")
test_model(imputed, model_choice, model, test_loader, seed, cohort)
print("> Defining longitudinal covariates model...")
long_model, long_criterion, long_optimizer = define_long_model(longitudinal_covariates.shape[2], hidden_size, num_layers, dropout, lr_long)
print("> Training longitudinal covariates model...")
long_model, long_test_loader = train_long_model(long_model, long_criterion, long_optimizer, longitudinal_covariates, y, seed, patience_long)
print("> Testing longitudinal covariates model performance metrics...")
test_long_model(long_model, long_test_loader)
print("------ COMPLETED ------\n")

print("--- EVALUATE MODELS ---")
print("> Generating scores using trained models...")
predictions = generate_predictions(model, long_model, X, y, is_missing, time_missing, static_covariates, longitudinal_covariates)
lr_fi = input("> Would you like to do LR feature importance? (y/n): ")
print("> Using logistic regression on embeddings...")
logistic_regression_embeddings(predictions, lr_fi)
km = input("> Would you like to generate Kaplan-Meier plots? (y/n): ")
if km == "y":
    print("> Generating cumulative incidence plots...")
    kmplots(predictions)
GRU_fi = input("> Would you like to do MaskedGRU feature importance? (y/n): ")
if GRU_fi == "y":
    print("> Generating MaskedGRU feature importance...")
    MaskedGRU_feature_importance(X, time_missing, is_missing, model)
print("------ COMPLETED ------\n")