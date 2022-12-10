import os
#import sys
#sys.path.append('C:/Users/lucbu/Documents/Master Thesis/Original Models/')
import glob
import shutil
import random as rnd
import numpy as np
import functools as ft
import h5py
import src.readData as readData
import src.preData as preData
import src.output as oo
import torch
import torch.nn as nn
import torch.nn.functional as torF
import sklearn.metrics as sklF
from src.classifiers import lgbm_model
from src.classifiers import xgb_model
from src.classifiers import nn_model
from src.classifiers import particleNet_model
from src.yuFunctional import npf2i
from sklearn.model_selection import train_test_split as tts
import pandas as pd

# g(f(x)) -> F(x, f, g...)
# g(f([x1, x2...])) -> FF([x1, x2...], f, g...)
def F(*z):
    z = list(z)
    return [*ft.reduce(lambda x, y: map(y, x), [z[:1]] + z[1:])][0]
FF = lambda *z: [*ft.reduce(lambda x, y: map(y, x), z)]
# f(x1, x2..., y1, y2...) -> fxy(f, x1, x2...)(y1, y2...)
# f(x1, x2..., y1, y2...) -> fyx(f, y1, y2...)(x1, x2...)
# f(x1, x2..., y1 = y1, y2 = y2...) -> fYx(f, y1 = y1, y2 = y2...)(x1, x2...)
fxy = lambda f, *x: lambda *y: f(*x, *y)
fyx = lambda f, *x: lambda *y: f(*y, *x)
fYx = lambda f, **x: lambda *y: f(*y, **x)

# remove the repeated letters in file names
# ".csvnone.h5" has 11 letters, .h5 has 3 letters
ftail = 3
data_cla = 2 # Number of classes

# Methods: xgb lgbm nn for new training 
# Methods: load_xgb load_nn load_lgbm to load saved models
# method = "lgbm" # so far the best seems to be lgbm
# method = "load_lgbm" # so far the best seems to be lgbm
method = "nn" # so far the best seems to be lgbm
# method = "load_nn" # so far the best seems to be lgbm
# method = "PN"
# method = "load_PN"

# sklearn fit
fit_para = {
    "early_stopping_rounds": 5000,
#   "eval_metric": "mlogloss",
    "eval_metric": "auc",
    "verbose": True
}

# LightGBM Classifier
lgbm_para = {
    "n_estimators": 50000,
    "first_metric_only": True,
#   For lgbm, cpu is even faster!
#   "device": "gpu", 
    "boosting_type": "gbdt",
    "num_leaves": 500,
    "max_depth": 15,
    "learning_rate": 1E-2,
    "subsample": 0.8,
    "objective": "binary",
    "n_jobs": os.cpu_count(),
    "silent": 1,
    "reg_alpha": 0.98,
    "reg_lambda": 0.98,
    "min_split_gain": 0,
    "min_child_weight": 0.7544,
#   "min_data_in_leaf": 99,
}

# Particle Net Classifier
pn_para = {
    "batch_size": 128,
    "epochs": 300,
    "early_stopping": 30,
    "learn_fn": torF.binary_cross_entropy,
    "optimizer": torch.optim.Adam,
    "scheduler": None,
#   "scheduler": fYx(
#       torch.optim.lr_scheduler.ExponentialLR,
#       gamma = 0.95, verbose = True
#   ),
    "metric": lambda y, x: sklF.roc_auc_score(x, y),
    "metric_better": '>',
    "learning_rate": 1E-3,
    "device": "cpu" if not torch.cuda.is_available() else "cuda",
    "pn_settings": {
        "embedding": True,
#       "conv_params": [
#           [16, [64, 64, 64]],
#           [16, [128, 128, 128]],
#           [16, [256, 256, 256]]
#       ],
#       "fc_params": [[0.1, 256]]
        "conv_params": [
#           [2, [16, 16, 16]],
#           [3, [8, 8, 8]]
            [2, [32, 32, 32]],
            [3, [16, 16, 16]]
        ],
        "fc_params": [[0.1, 128]]
    }
}

# NN Classifier
nn_layers = (514,432,289,320,1017,53,16,249,76,91,85,145,37,1)
#nn_layers = (128, 256, 128, 64, 32, 1)
nn_para = {
    "batch_size": 32,
    "epochs": 100,
    "early_stopping": 10,
    "learn_fn": torF.binary_cross_entropy,
    "optimizer": torch.optim.Adam,
#   "scheduler": None,
    "scheduler": fYx(
        torch.optim.lr_scheduler.ExponentialLR,
        gamma = 0.95, verbose = True
    ),
    "metric": lambda y, x: sklF.roc_auc_score(x, y),
    "metric_better": '>',
    "learning_rate": 0.0003,
    "device": "cpu" if not torch.cuda.is_available() else "cuda",
    "network": [
# These 2 layers are added when calling
####### nn.BatchNorm1d(input_dim)
####### nn.Linear(input_dim, nn_layers[0])
        nn.Dropout(0.5),
        nn.ReLU(),
        nn.Linear(nn_layers[0], nn_layers[1]),
        nn.Dropout(0.5),
        nn.ReLU(),
        nn.Linear(nn_layers[1], nn_layers[2]),
        nn.Dropout(0.5),
        nn.ReLU(),
        nn.Linear(nn_layers[2], nn_layers[3]),
        nn.ReLU(),
        nn.Linear(nn_layers[3], nn_layers[4]),
        nn.ReLU(),
        nn.Linear(nn_layers[4], nn_layers[5]),
        nn.ReLU(),
        nn.Linear(nn_layers[5], nn_layers[6]),
        nn.ReLU(),
        nn.Linear(nn_layers[6], nn_layers[7]),
        nn.ReLU(),
        nn.Linear(nn_layers[7], nn_layers[8]),
        nn.ReLU(),
        nn.Linear(nn_layers[8], nn_layers[9]),
        nn.ReLU(),
        nn.Linear(nn_layers[9], nn_layers[10]),
        nn.ReLU(),
        nn.Linear(nn_layers[10], nn_layers[11]),
        nn.ReLU(),
        nn.Linear(nn_layers[11], nn_layers[12]),
        nn.ReLU(),
        nn.Linear(nn_layers[12], nn_layers[13]),
        nn.Sigmoid()
    ]
}

# XGBoost Classifier
xgb_para = {
    "n_estimators": 5000,
    "tree_method": "gpu_hist",
    "predictor": "gpu_predictor",
    "nthread": os.cpu_count(),
#   "num_class": data_cla,
#   "objective": "multi:softmax",
    "objective": "binary:logistic",
    "verbosity": 1,
    "max_depth": 15,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
#   "min_child_weight": 5,
    "learning_rate": 1E-2
}

if method == "xgb" or method == "nn":
    print("cpu" if not torch.cuda.is_available() else "cuda")

homePath = "C:/Users/lucbu/Documents/Master Thesis/Original Models/"
dataPath = "data/"
modelPath = "models/"

#hf = h5py.File(homePath + dataPath + "4top_lep_400k.h5", 'r')
#reg = np.array(hf.get('X')).astype("float32")
#sig = np.array(hf.get('y')).astype("int8")
#reg, sig = preData.permutation(reg, sig)
#reg_train, reg_temp, sig_train, sig_temp = tts(
#    reg, sig, test_size = 0.2
#)
#reg_val, reg_test, sig_val, sig_test = tts(
#    reg_temp, sig_temp, test_size = 0.5
#)


def dataATLAS():
     hf = pd.read_csv("C:/Users/lucbu/Documents/Master Thesis/Data Polina/ntuples_MC16ade_4tops_all_bkg_for_Luc.csv", header = 0, sep=',')
     hf = hf.fillna(0)  
        
     hf['ProcessID'] = hf['ProcessID'].where(hf['ProcessID'] == '4tops', 0)
     hf['ProcessID'] = hf['ProcessID'].where(hf['ProcessID'] != '4tops', 1)
      
     tops = hf[hf['ProcessID']==1]
     background = hf[hf['ProcessID']==0]
    
     sigweights = tops['evtweight'].to_numpy()
     bkgweights = background['evtweight'].to_numpy()
    
     sigweights = 0.5*(sigweights * (1/np.sum(sigweights)))
     bkgweights = 0.5*(bkgweights * (1/np.sum(bkgweights)))
    
     w_norm = np.hstack((sigweights,bkgweights))
    
     hf = tops.append(background)
    
     variables = hf.columns[3:].tolist()
    
     x = hf[variables].to_numpy()
     y = hf['ProcessID'].to_numpy()
    
     perm = np.random.permutation(x.shape[0])
     x = x[perm]
     y = y[perm]
     w_norm = w_norm[perm]
    
     x = x.astype("float32")
     y = y.astype("float32")
     w_norm = w_norm.astype("float32")
    
     x_train, x_test, w_train, w_test, y_train, y_test = tts(x, w_norm, y, train_size=0.9)
     x_train, x_val, y_train, y_val = tts(x_train, y_train, train_size=0.89)
     
     print(x_train.shape)
     print(y_train.shape)
     print(x_val.shape)
     print(y_val.shape)
     print(x_test.shape)
     print(y_test.shape) 
     
     return x_train, x_test, w_train, w_test, y_train, y_test

def dataRoberto():
    hf = h5py.File("C:/Users/lucbu/Documents/Master Thesis/Data Roberto/4tops_fcn_splitted.h5", 'r')
    reg_train = np.array(hf.get("X_train")).astype("float32")
    sig_train = np.array(hf.get("Y_train")).astype("int8")

    reg_val = np.array(hf.get("X_val")).astype("float32")
    sig_val = np.array(hf.get("y_val")).astype("int8")
    
    reg_test = np.array(hf.get("X_test")).astype("float32")
    sig_test = np.array(hf.get("y_test")).astype("int8")
    #reg_test, sig_test = preData.permutation(reg_test, sig_test)
    print(reg_train.shape)
    print(sig_train.shape)
    print(reg_val.shape)
    print(sig_val.shape)
    print(reg_test.shape)
    print(sig_test.shape)   
    return reg_train, reg_val, reg_test, sig_train, sig_val, sig_test


reg_train, reg_val, reg_test, sig_train, sig_val, sig_test = dataRoberto()
#reg_train, reg_val, reg_test, sig_train, sig_val, sig_test = dataATLAS()
print(sig_test[0:50])


## Train
os.chdir(homePath)
if not os.path.isdir("models"):
    os.mkdir("models")
# XGBoost Model
if method == "xgb":
    model = xgb_model(
        reg_train, reg_val, sig_train, sig_val,
        xgb_para, fit_para
    ).train().save(modelPath + "xgb" + ".model")
# LightGBM Model
if method == "lgbm":
    model = lgbm_model(
        reg_train, reg_val, sig_train, sig_val,
        lgbm_para, fit_para
    ).train().save(modelPath + "lgbm" + ".model")
# NN Model
if method == "nn":
    model = nn_model(
        reg_train, reg_val, sig_train, sig_val,
        nn_layers, nn_para
    ).train().save(modelPath + "nn" + ".model")
# PN Model
if method == "PN":
    model = particleNet_model(
        reg_train, reg_val, sig_train, sig_val, pn_para
    ).train().save(modelPath + "pn" + ".model")
# Load
if method == "load_xgb":
    model = xgb_model().load(modelPath + "xgb" + ".model")
if method == "load_lgbm":
    model = lgbm_model().load(modelPath + "lgbm" + ".model")
if method == "load_nn":
    model = nn_model().load(modelPath + "nn" + ".model")
if method == "load_PN":
    model = particleNet_model(pn_para = pn_para).load(
        modelPath + "pn" + ".model"
    )

#y_pred_prob = model.pred_prob(reg_test)
#np.savetxt('C:/Users/lucbu/Documents/Master Thesis/Results/scores/FCN.csv', y_pred_prob)
oo.output((reg_test, sig_test), model)
oo.draw_roc((reg_test, sig_test), model,
            method[5:] if method[:4] == "load" else method,
            method[5:] if method[:4] == "load" else method)
