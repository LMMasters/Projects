# TensorFlow and tf.keras
import tensorflow as tf



import keras
import keras.backend
from keras import layers
from keras import models
from keras import optimizers
import keras.utils

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import sklearn
from sklearn.utils import shuffle

import os
import pickle

def build_DNN(X_train):
    input2 = [layers.Input(shape = (len(X_train[i][0]),)) for i in range(0,len(X_train))]
    if len(input2) == 1:
        x = layers.Dense(64,activation='relu')(input2[0])
    else:
        x = layers.concatenate(inputs = input2, axis = -1)
        x = layers.Dense(64, activation = 'relu')(x)
    x = layers.Dense(128, activation = 'relu')(x)
    output = layers.Dense(2, activation = 'softmax')(x)
    model = models.Model(inputs = input2, 
                         outputs = output)
    model.compile(loss = 'categorical_crossentropy',
                optimizer = 'adam',
                metrics = ['categorical_crossentropy', 'accuracy'])
    return model

def build_CNN_2D(X_train, grid):
    input2 = [layers.Input(shape = (len(X_train[i][0]),)) for i in range(1,len(X_train))]
    input1 = layers.Input(shape = (grid, grid,1))
    x = layers.Conv2D(32, (5, 5), activation = 'relu', padding = 'same')(input1)
    x = layers.Conv2D(32, (3, 3), activation = 'relu', padding = 'same')(x)
    x = layers.Conv2D(32, (2, 2), activation = 'relu', padding = 'same')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.MaxPool2D((2, 2))(x)
    x1 = layers.Flatten()(x)
    if input2 == []:
        x = layers.Dense(64, activation='relu', name = 'dense1')(x1)
    else:
        x = layers.concatenate(inputs = [x1] + input2, axis = -1)
        x = layers.Dense(64, activation='relu', name = 'dense2')(x)
    x = layers.Dense(128, activation='relu', name = 'dense3')(x)
    output = layers.Dense(2, activation='softmax')(x)
    model = models.Model(inputs= [input1] + input2,
                         outputs = output)
    model.compile(loss = 'categorical_crossentropy',
                optimizer = 'adam',
                metrics = ['categorical_crossentropy', 'accuracy'])
    return model

def build_CNN_1D(n, nex, input_size):
    
    # n = total number of input features
    # nex = number of expert features (xaugs)
    
    inpts = []
    xaugs = []
    xlayers = []
    
    # loop over all input variables
    for i in range(n):
        
        # particle list inputs 
        if(i < n-nex):

            inpt = layers.Input(shape = (input_size,1))

            x = layers.Conv1D(16, 2, padding = 'same', activation='relu')(inpt)
            x = layers.Conv1D(8, 2, padding = 'same', activation='relu')(x)
            x = layers.Conv1D(16, 2, padding = 'same', activation='relu')(x)
            x = layers.Conv1D(32, 5, padding = 'same', activation='relu')(x)
            x = layers.Dropout(0.2)(x)
            x = layers.MaxPool1D(2)(x)
            x = layers.Conv1D(8, 3, padding = 'same', activation='relu')(x)
            x = layers.Conv1D(32, 3, padding = 'same', activation='relu')(x)
            x = layers.Dropout(0.2)(x)
            x = layers.MaxPool1D()(x)
            x1 = layers.Flatten()(x)

            inpts.append(inpt)
            xlayers.append(x1)
        
        
        # expert variable inputs 
        elif((nex > 0)):

            inpt = layers.Input(shape = (1,))
            xaugs.append(inpt)
    
    #concatenation of particle list inputs with expert variable inputs
    if(n > 1):
        x = layers.concatenate(inputs=xlayers+xaugs, axis=-1)
        
    '''
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(432, activation='relu')(x)
    x = layers.Dense(289, activation='relu')(x)
    x = layers.Dense(320, activation='relu')(x)
    x = layers.Dense(1017, activation='relu')(x)
    x = layers.Dense(53, activation='relu')(x)
    x = layers.Dense(16, activation='relu')(x)
    x = layers.Dense(249, activation='relu')(x)
    x = layers.Dense(76, activation='relu')(x)
    x = layers.Dense(91, activation='relu')(x)
    x = layers.Dense(85, activation='relu')(x)
    x = layers.Dense(145, activation='relu')(x)
    x = layers.Dense(37, activation='relu')(x)
    '''

    #x = layers.Dense(64, activation='relu')(x)
    #x = layers.Dense(128, activation='relu')(x)
    
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(16, activation='relu')(x)
   
    output = layers.Dense(2, activation='softmax')(x) 
    model = models.Model(inputs=inpts+xaugs, outputs=output)
    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['categorical_crossentropy', 'AUC'])
    
    return model

#def build_CNN_1D(n, nex, input_size):
#    
#    # n = total number of input features
#    # nex = number of expert features (xaugs)
#    
#    inpts = []
#    xaugs = []
#    xlayers = []
#    
#    # loop over all input variables
#    for i in range(n):
#        
#        # particle list inputs 
#        if(i < n-nex):
#
#            inpt = layers.Input(shape = (input_size,1))
#
#            x = layers.Conv1D(64, 3, padding = 'same', activation='relu')(inpt)
#            x = layers.Conv1D(64, 1, padding = 'same', activation='relu')(x)
#            x = layers.Dropout(0.2)(x)
#            x = layers.MaxPool1D(2)(x)
#            x = layers.Conv1D(32, 3, padding = 'same', activation='relu')(x)
#            x = layers.Conv1D(32, 1, padding = 'same', activation='relu')(x)
#            x = layers.Dropout(0.2)(x)
#            x = layers.MaxPool1D()(x)
#            x1 = layers.Flatten()(x)
#
#            inpts.append(inpt)
#            xlayers.append(x1)
#        
#        
#        # expert variable inputs 
#        elif((nex > 0)):
#
#            inpt = layers.Input(shape = (1,))
#            xaugs.append(inpt)
#    
#    #concatenation of particle list inputs with expert variable inputs
#    if(n > 1):
#        x = layers.concatenate(inputs=xlayers+xaugs, axis=-1)
#
#    x = layers.Dense(64, activation='relu')(x)
#    x = layers.Dense(128, activation='relu')(x)
#    
#    output = layers.Dense(2, activation='softmax')(x) 
#    model = models.Model(inputs=inpts+xaugs, outputs=output)
#    model.compile(loss='categorical_crossentropy',
#                optimizer='adam',
#                metrics=['categorical_crossentropy', 'accuracy'])
#    
#    return model


def run_DNN(CNN, X_train, Y_train, checkpoint_path):
    model_checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_path, monitor = 'val_loss', verbose = 1, save_best_only = True, save_weights_only = False, mode = 'auto', period = 1)    
    EPOCHS = 60
    early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 10, min_delta = 1e-5)
    history = CNN.fit(
        X_train, Y_train,
        epochs = EPOCHS,
        validation_split = 0.2,
        verbose = 1,
        callbacks = [early_stop, model_checkpoint])

def build_XY(features, label, dic1, dic2, signal="Zbb"):
    X = [np.concatenate((dic1[key], dic2[key])) for key in features]
    Y = [np.concatenate((dic1[key], dic2[key])) for key in label]
    dim = [ele.shape+(1,) for ele in X]
    for i in range(0,len(features)):
        X[i] = X[i].reshape(dim[i])

    #choose the correct column for signal labels                               
    if "Zbb" in signal:
        Y = [Y[0][:,::2]]
    else:
        Y = [Y[0][:,:2]]
        
    #Randomize! 
    seed = 1234567890
    prng = np.random.RandomState(seed)
    # split signal and background indices                                      
    ind = np.argwhere(Y[0])[:,0]
    # shuffle indices randomly                                                 
    prng.shuffle(ind)
    for j in range(0,len(features)):
        X[j] = X[j][ind]
    Y = [Y[0][ind]]

    return X,Y


def get_feat(xaugs=True, flav=True,  images=False, particleInfo=False):
    
    feat = []

    if images:
        feat.insert(0,'jetImages')

    if particleInfo:
        feat.extend(['jetconstPt_log',
                     'jetconstEta_abs',
                     'jetconstE_log',
                     'jetconstPt_Jetlog',
                     'charge',
                     'isEle',
                     'isPho',
                     'isMuon',
                     'isCh',
                     'isNh',
                     'delta_eta',
                     'delta_phi',
                     'deltaR_jet',
                     'deltaR_subjet0',
                     'deltaR_subjet1',
                     'dxy',
                     'dz'])

    if(xaugs):
        feat.extend(['jetPt',
                     'jetEta',
                     'jetPhi',
                     'jetMass',
                     'jetMassSD',
                     'deltaR_subjets',
                     'z',
                     'tau1_b05',
                     'tau2_b05',
                     'tau3_b05',
                     'tau1_sd_b05',
                     'tau2_sd_b05',
                     'tau3_sd_b05',
                     'tau1_b10',
                     'tau2_b10',
                     'tau3_b10',
                     'tau1_sd_b10',
                     'tau2_sd_b10',
                     'tau3_sd_b10',
                     'tau1_b20',
                     'tau2_b20',
                     'tau3_b20',
                     'tau1_sd_b20',
                     'tau2_sd_b20',
                     'tau3_sd_b20',
                     'beta3',
                     'beta3_sd',
                     'tau21',
                     'jetpull_abs'])

    if flav:
        feat.extend(['chMult',
                     'neutMult',
                     'phoMult',
                     'eleMult',
                     'muMult',
                     'dxy_max',
                     'dz_max'])
    return feat

