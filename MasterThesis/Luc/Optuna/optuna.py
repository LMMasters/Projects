#import sys
#sys.path.append('C:/Users/lucbu/Documents/Master Thesis/Zhongyi/')

import os
import argparse
import numpy as np
import h5py

import optuna

from goFish2 import zhongyi_funcs
from particle_transformer.train import rob_funcs

save_best_file = "C:/Users/lucbu/Documents/Master Thesis/Zhongyi/models/best_MLP.txt"
data_loc = "C:/Users/lucbu/Documents/Master Thesis/Zhongyi/data/4tops_splitted.h5"

n_trials = 100 #Amount of optimization trials
method = "MLP" 
# method = "LGBM"
# method = "PN"
# method = "PTrans"

if method == "MLP":
    def objective(trial):  
        #Optimizable variables
        batch_size = trial.suggest_int("batch_size", 16, 2048, log=True)
        epochs = 50
        early_stopping = 10
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        n_layers = trial.suggest_int("n_layers", 5, 20)
        num_hidden = []
        for i in range(n_layers):
            num_hidden.append(trial.suggest_int("n_units_l{}".format(i), 4, 1024, log=True))
        
        #Get a score for a trial
        score = zhongyi_funcs(trial, "nn", data_loc, batch_size=batch_size, epochs=epochs, early_stopping=early_stopping, lr=lr, n_layers=n_layers, num_hidden=num_hidden)
        return score
    
if method == "LGBM":
    def objective(trial):  
        #Optimizable variables
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        n_estimators = trial.suggest_int("n_estimators", 10000, 100000, log=True)
        num_leaves = trial.suggest_int("num_leaves", 100, 1000, log=True)
        max_depth = trial.suggest_int("max_depth", 5, 20)
        subsample = trial.suggest_float("subsample", 0.5, 0.9)
        reg_alpha = trial.suggest_float("reg_alpha", 0.95, 0.99)
        reg_lambda = trial.suggest_float("reg_lambda", 0.95, 0.99)
        min_split_gain = trial.suggest_float("min_split_gain", 0, 0.1)
        min_child_weight = trial.suggest_float("min_child_weight", 0.7, 0.8)
        
        #Get a score for a trial
        score = zhongyi_funcs(trial, "lgbm", data_loc, lr=lr, n_estimators=n_estimators, num_leaves=num_leaves, max_depth=max_depth, subsample=subsample, reg_alpha=reg_alpha, reg_lambda=reg_lambda, min_split_gain=min_split_gain, min_child_weight=min_child_weight)
        return score

if method == "PN":
    def objective(trial):  
        #Optimizable variables
        batch_size = trial.suggest_int("batch_size", 16, 2048, log=True)
        epochs = 50
        early_stopping = 10
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        conv_params_choice = trial.suggest_categorical("conv_params", ["Choice A", "Choice B"]) #More choices can be added
        if conv_params_choice == "Choice A":
            conv_params = [
                [2, [32, 32, 32]],
                [3, [16, 16, 16]]
                ]
        if conv_params_choice == "Choice B":
            conv_params = [
                [2, [16, 16, 16]],
                [3, [8, 8, 8]]
                ]
        
        #Get a score for a trial
        score = zhongyi_funcs(trial, "PN", data_loc, batch_size=batch_size, epochs=epochs, early_stopping=early_stopping, lr=lr, conv_params=conv_params)
        return score

if method == "PTrans":
    def objective(trial):  
        #Optimizable variables
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        lr_patience = 10
        n_epochs = 100
        n_batches = 100
        input_dim = 10
        num_classes = 2
        aux_dim = 8
        fc_params_choice = trial.suggest_categorical("fc_params", ["Choice A", "Choice B"]) #More choice can be added
        if fc_params_choice == "Choice A":
            fc_params = [[128, 0.1], [512, 0.1], [128, 0.1]]
        if fc_params_choice == "Choice B":
            fc_params = [[1024, 0.1], [512, 0.1], [128, 0.1]]

        #Get a score for a trial
        score = rob_funcs(trial, data_loc, lr=lr, lr_patience=lr_patience, n_epochs=n_epochs, n_batches=n_batches, input_dim=input_dim, num_classes=num_classes, aux_dim=aux_dim, fc_params=fc_params)
        return score
    
#Optimize
if method == "PTrans":
    study = optuna.create_study(direction="minimize")

else:    
    study = optuna.create_study(direction="maximize")
    
study.optimize(objective, n_trials)


#Print results and save to file
print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
    
#Save results
with open(save_best_file, "w") as hf:
     hf.write("Value: {} \n \n".format(trial.value))
     hf.write("Params: \n")
     for key, value in trial.params.items():
         hf.write("   {}: {} \n".format(key, value))
