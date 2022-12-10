# TensorFlow and tf.keras
import tensorflow as tf
#tf.enable_eager_execution()

from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd

import keras
import keras.backend
from keras import layers
from keras import models
import keras.utils

# Helper libraries
import numpy as np
from scipy import stats,special

from matplotlib.colors import LogNorm
from matplotlib import gridspec
import math
import time
import h5py
import sklearn
from sklearn.utils import shuffle

# innvestigate for lrp
import innvestigate
import innvestigate.utils as iutils
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib as mpl

import os
import datetime

feature_names={ 'jetImages':'Image',
                'particleList':'List',
                'jetconstE_log':'log($E$)',
                'jetconstEta_abs':'$|\eta|$',
                'jetconstPt_Jetlog':'log($p_T / p_{T_{jet}}$)',
                'jetconstPt_log':'log($p_T$)',
                'isPho':r'$b_{\gamma}$',
                'isNh':r'$b_{NH}$',
                'isMuon':r'$b_{\mu}$',
                'isEle':r'$b_{e}$',
                'isCh':r'$b_{CH}$',
                'charge':r'$q$',
                'jetPt':r'$p_{T,jet}$',
                'jetEta':r'$\eta(jet)$',
                'jetPhi':r'$\phi(jet)$',
                'jetMass':r'$m_{jet}$',
                'jetMassSD':r'$m_{jet,sd}$',
                'tau1_b05':r'$\tau_{1}^{(0.5)}$',
                'tau2_b05':r'$\tau_{2}^{(0.5)}$',
                'tau3_b05':r'$\tau_{3}^{(0.5)}$',
                'tau1_sd_b05':r'$\tau_{1,sd}^{(0.5)}$',
                'tau2_sd_b05':r'$\tau_{2,sd}^{(0.5)}$',
                'tau3_sd_b05':r'$\tau_{3,sd}^{(0.5)}$',
                'tau1_b10':r'$\tau_{1}^{(1)}$',
                'tau2_b10':r'$\tau_{2}^{(1)}$',
                'tau3_b10':r'$\tau_{3}^{(1)}$',
                'tau1_sd_b10':r'$\tau_{1,sd}^{(1)}$',
                'tau2_sd_b10':r'$\tau_{2,sd}^{(1)}$',
                'tau3_sd_b10':r'$\tau_{3,sd}^{(1)}$',
                'tau1_b15':r'$\tau_{1}^{(1.5)}$',
                'tau2_b15':r'$\tau_{2}^{(1.5)}$',
                'tau3_b15':r'$\tau_{3}^{(1.5)}$',
                'tau1_sd_b15':r'$\tau_{1,sd}^{(1.5)}$',
                'tau2_sd_b15':r'$\tau_{2,sd}^{(1.5)}$',
                'tau3_sd_b15':r'$\tau_{3,sd}^{(1.5)}$',
                'tau1_b20':r'$\tau_{1}^{(2)}$',
                'tau2_b20':r'$\tau_{2}^{(2)}$',
                'tau3_b20':r'$\tau_{3}^{(2)}$',
                'tau1_sd_b20':r'$\tau_{1,sd}^{(2)}$',
                'tau2_sd_b20':r'$\tau_{2,sd}^{(2)}$',
                'tau3_sd_b20':r'$\tau_{3,sd}^{(2)}$',
                'chMult':r'$N_{ch}$',
                'neutMult':r'$N_{neut}$',
                'phoMult':r'$N_{\gamma}$',
                'eleMult':r'$N_{e}$',
                'muMult':r'$N_{\mu}$',
                'jetpull':r'$\phi_{pull}$',
                'jetpull_abs':r'|$\phi_{pull}$|',
                'beta3':r'$\beta_{3}$',
                'beta3_sd':r'$\beta_{3}^{g}$',
                'tau21':r'$\tau_{2}^{(1)} / \tau_{1}^{(1)}$',
                'deltaR_jet':'$\Delta R$(jet)',
                'deltaR_subjet0':'$\Delta R$(subjet0)',
                'deltaR_subjet1':'$\Delta R$(subjet1)',
                'deltaR_subjets':'$\Delta R_{subjets}$',
                'delta_eta':'$\Delta \eta$',
                'delta_phi':'$\Delta \phi$',
                'z':r'$z$',
                'dxy_max':r'$d_{xy,max}$',
                'dz_max':r'$d_{z,max}$',
                'dxy':'$d_{xy}$',
                'dz':'$d_{z}$',
              }


def printTime(delta, definition='Elapsed Time'):
    
    
    time_string = '{0:0.1f} h {1:0.1f} min {2:0.2f} sec'.format(delta // 3600, delta % 3600 // 60, delta % 3600 % 60)
    full_string = definition + ': ' + time_string
    print(full_string)
    


def get_lrp_score(test_inputs, model, nXvar, totalVar, batchShape=[20]):
    
    
    # test_inputs is X_test output from build_XY function in networkBuilder.py 
    # model loaded from model = keras.models.load_model(model_name)
    # nXvar = number of expert features (xaugs)
    # totalVar = total number of features
    # batchShape =  [20] for particle list: number of constituent particles per event
    #               [16, 16] for jet images: grid shape
    
    lrp_toc = time.time()
    
    # rename dense layers for model_wo_softmax function
    layer_i = 1
    for layer in model.layers:
        if('dense' in layer.name):
            layer.name = 'renamed_dense_layer'+str(layer_i)
            layer_i += 1
    
    # strip model of softmax layer
    best_model_wo_softmax = innvestigate.utils.keras.graph.model_wo_softmax(model)
     
    # build lrp analyzer using Preset A: LRP_alpha1beta0 for conv layers and LRP_epsilon for dense layers
    lrp_analyzer = innvestigate.analyzer.relevance_based.relevance_analyzer.LRPSequentialPresetA(best_model_wo_softmax)
   
    # run lrp analysis in chunks so that kernel does not die
    nElem = 10000
    batchsize = test_inputs[0].shape[0]
    slices = batchsize // nElem
    size_from_loop = batchsize - (batchsize % nElem)
    
    print()
    print('Running LRP Analysis')
    print('Total Events: {0:0.0f}'.format(batchsize))
    print('Split into chunk size: {0:0.0f}'.format(nElem))
    print()
    
    lrp_list = []
    

    # get lrp scores from all chunks except last chunk
    for i in range(slices):

        lrp_i = lrp_analyzer.analyze([element[nElem*i:nElem*(i+1)] for element in test_inputs])
        lrp_list.append(lrp_i)
        del lrp_i

    
    # get lrp scores from last chunk
    lrp_list_last = lrp_analyzer.analyze([element[size_from_loop:] for element in test_inputs])
    
    # combine chunks into separate arrays for particle list features (lrp_plist) and xaug features (lrp_xaugs)
    lrp_plist_list = []
    lrp_xaugs_list = []
    
    lrp_plist_list_last = []
    lrp_xaugs_list_last = []


    # append lrp scores to lrp_plist or lrp_xaugs
    for i in range(totalVar):
        for s in range(slices):
            if(i < (totalVar-nXvar)):
                lrp_plist_list.append(lrp_list[s][i])
                    
            else:
                lrp_xaugs_list.append(lrp_list[s][i])
                
    # append lrp scores from last chunk         
    for i in range(totalVar):
        if(i < (totalVar-nXvar)):
            lrp_plist_list_last.append(lrp_list_last[i])

        else:
            lrp_xaugs_list_last.append(lrp_list_last[i])
                
                
    # copy to arrays           
    lrp_plist_first = np.array(lrp_plist_list).reshape(totalVar-nXvar, size_from_loop, *batchShape, 1)
    lrp_xaugs_first = np.array(lrp_xaugs_list).reshape(nXvar, size_from_loop, 1)
    
    lrp_plist_last = np.array(lrp_plist_list_last).reshape(totalVar-nXvar,(batchsize % nElem),*batchShape,1)
    lrp_xaugs_last = np.array(lrp_xaugs_list_last).reshape(nXvar,(batchsize % nElem),1)
                
    
    # append last chunk to lrp scores arrays
    lrp_plist = np.append(lrp_plist_first, lrp_plist_last, axis=1)
    lrp_xaugs = np.append(lrp_xaugs_first, lrp_xaugs_last, axis=1)

    del lrp_plist_list
    del lrp_xaugs_list
    del lrp_plist_list_last
    del lrp_xaugs_list_last
    del lrp_list
    del lrp_plist_first
    del lrp_xaugs_first
    del lrp_plist_last
    del lrp_xaugs_last
    
    lrp_tic = time.time()
    
    print()
    printTime(lrp_tic - lrp_toc, 'LRP Analysis Time')
    
    return lrp_plist, lrp_xaugs
    

def get_normalized_lrp_score(nXvar, totalVar, lrp=[], lrp_xaugs=[], batchShape=[20]):
    
    # lrp = list of images array or list of particle list features array
    #       output of get_lrp_score()
    # lrp_xaugs = list of expert variable features array
    #             output of get_lrp_score()
    # nXvar = number of expert features (xaugs)
    # totalVar = total number of features
    # batchShape =  [20] for particle list: number of constituent particles per event
    #               [16, 16] for jet images: grid shape
    
    
    lrp_axis = (0,2,3)
    if(len(batchShape)==2):
        lrp_axis = (0,2,3,4)
        
        
    if((len(lrp_xaugs) == 0) and (len(lrp) == 0)):
        print('Both arrays are empty')
        return [], []
    
    
    if(len(lrp_xaugs) > 0 and (len(lrp) > 0)):
        
        maxval_lrp = np.max(np.abs(lrp), axis=(lrp_axis))
        maxval_xaugs = np.max(np.abs(lrp_xaugs), axis=(0,2))

        maxval = np.where(maxval_lrp >= maxval_xaugs, maxval_lrp, maxval_xaugs)
        
        maxval_repeated_xaugs = np.tile(maxval, nXvar).reshape(lrp_xaugs.shape)
        maxval_repeated_lrp = np.tile(np.repeat(maxval, np.prod(batchShape), axis =0), totalVar-nXvar).reshape(lrp.shape)

        lrp_xaugs_norm = np.where(maxval_repeated_xaugs > 0, lrp_xaugs / maxval_repeated_xaugs, 0)
        lrp_norm = np.where(maxval_repeated_lrp > 0, lrp / maxval_repeated_lrp, 0)

    elif((len(lrp) > 0) and not (len(lrp_xaugs) > 0)):    
    
        maxval = np.max(np.abs(lrp), axis=(lrp_axis))    #maxval_lrp
        maxval_repeated_lrp = np.tile(np.repeat(maxval, np.prod(batchShape), axis =0), totalVar-nXvar).reshape(lrp.shape)
        lrp_norm = np.where(maxval_repeated_lrp > 0, lrp / maxval_repeated_lrp, 0)      
        lrp_xaugs_norm = []
        
        
    elif((len(lrp_xaugs) > 0) and not ((len(lrp) > 0))):  

        maxval = np.max(np.abs(lrp_xaugs), axis=(0,2))  #maxval_xaugs
        maxval_repeated_xaugs = np.tile(maxval, nXvar).reshape(lrp_xaugs.shape)
        lrp_xaugs_norm = np.where(maxval_repeated_xaugs > 0, lrp_xaugs / maxval_repeated_xaugs, 0)
        lrp_norm = []
    
 
    return lrp_norm, lrp_xaugs_norm



def get_mean_relevance(lrp_models=[], lrp_xaugs_models=[], batchShape=[20], isSum=True):
    
    # lrp_models = list of lrp (images or particle list) score arrays to average over
    # lrp_xaugs_models = list of xaug lrp scores
    # batchShape =  [20] for particle list: number of constituent particles per event
    #               [16, 16] for jet images: grid shape
    # isSum = True or False : take sum of relevance of image grid pixels (True) or take maximum relevance (False)
    
    
    
    lrp_mean = []
    lrp_xaugs_mean = []
    lrp_std = []
    lrp_xaugs_std = []
    
    lrp_axis = (0,2,3)
    if(len(batchShape)==2):
        lrp_axis = (0,2,3)
    

    if((len(lrp_models) > 0)):
        
        if ((len(batchShape)==2) and (isSum)):
            lrp_models = np.sum(np.abs(lrp_models),axis = (3,4))  # sum only along image grid
            
            # average by nonzero pixels
            
#             lrp_models = np.divide(np.sum(lrp_models,axis = (3,4)),  np.count_nonzero( lrp_models, axis = (3,4) ))
            
        elif ((len(batchShape)==2) and not(isSum)):
            lrp_models = np.max(lrp_models,axis = (3,4)) # max only along image grid
            
            
        if ((len(batchShape)==1) and (isSum)):
            
            lrp_models = np.sum(lrp_models,axis = 3)  # sum particle list lrp
          
        elif ((len(batchShape)==1) and not(isSum)):

            lrp_models = np.max(lrp_models,axis = 3) # max only along image grid    
        
        
        nEvents = lrp_models[0].shape[1]
        
        #print(lrp_models[0].shape, nEvents)

        lrp_means = np.sum(np.abs(lrp_models), axis=lrp_axis[1:]) / (nEvents)


#         if(len(batchShape)==2):
#             lrp_means = np.sum(np.abs(lrp_models), axis=lrp_axis[1:]) / (nEvents)
#         else:
#             lrp_means = np.sum(np.abs(lrp_models), axis=lrp_axis[1:]) / (np.prod(batchShape)*nEvents)
        lrp_mean = np.sum(lrp_means, axis=0) / (len(lrp_models))
        lrp_std = np.std(lrp_means, axis=0)
        
    
    if(len(lrp_xaugs_models) > 0):
        
        nEvents = lrp_xaugs_models[0].shape[1]
        lrp_xaugs_means = np.sum(np.abs(lrp_xaugs_models), axis=(2,3)) / nEvents
        lrp_xaugs_mean = np.sum(lrp_xaugs_means, axis=0) / len(lrp_xaugs_models)
        lrp_xaugs_std = np.std(lrp_xaugs_means, axis=0)
        
    if(isSum):
        lrp_mean_list = [np.sum(lrp_mean)] + [lrp for lrp in lrp_xaugs_mean]
        lrp_std_list = [np.std(lrp_means)] + [lrp for lrp in lrp_xaugs_std]
        
    else:
        lrp_mean_list = [lrp for lrp in lrp_mean] + [lrp for lrp in lrp_xaugs_mean]
        lrp_std_list = [std for std in lrp_std] + [lrp for lrp in lrp_xaugs_std]


    return lrp_mean_list, lrp_std_list




def make_relevance_bar_plot(features, LRP_mean, LRP_std, topN=None):
    
    # features = list of features in same order as LRP_mean
    # LRP_mean = array of averaged relevance scores
    # LRP_std = standard deviation of averaged relevance scores
    # topN = number of highest relevance scores to plot (for all features, topN=None)
    
    
    params = {'legend.fontsize': 'x-large',
     'axes.labelsize': 'x-large',
     'axes.titlesize':'x-large',
     'font.family':'serif',
     'xtick.labelsize':'x-large',
     'ytick.labelsize':'x-large',
     'figure.facecolor':'white',
     'axes.grid':True,
     'grid.alpha':1.0 
         }
    
    plt.rcParams.update(params)
    plt.style.context('default')
    mpl.rcParams.update({'hatch.linewidth':2.0})
    
    size_for_all_features = [15,10]
    size_for_5_features = [7.5,5]
    size = size_for_all_features
    fs = 20 # fontsize of y-label and y-ticks
    
    if(topN != None):
        if(topN < 20):
            size = size_for_5_features
            fs = 15 # fontsize of y-label and y-ticks
        
    
    fig = plt.figure(figsize=size )
    
    LRP_order = [features[i] for i in np.argsort(LRP_mean)]
    LRP_std_sorted = [LRP_std[i] for i in np.argsort(LRP_mean)]
    LRP_order2 = [feature_names[feat] for feat in LRP_order]
    relevances = np.sort(LRP_mean)    
    
    barplot = plt.bar(LRP_order2, relevances)
    
    xmax = len(LRP_order2)-0.5
    xmin=-0.5
    if(topN==None):
        xmin=-0.5
    else:
        xmin = xmax-topN
    
    ymin = 0
    ymax = (relevances[-1] + LRP_std_sorted[-1])*1.02
    
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    
    plt.xticks(rotation=90, fontsize=15)
    plt.yticks(fontsize=fs)
    
    plt.ylabel('Mean Normalized Relevance', fontsize=fs)

    
    count = 0
    for b in barplot:    
        plt.fill_between([b.get_x(), b.get_x() + b.get_width()],
                         b.get_height() - LRP_std_sorted[count],
                         b.get_height() + LRP_std_sorted[count],
                         lw=0,
                         fc='none',
                         ec='k',
                         hatch='//',
                         zorder=2)    
        count += 1
        
        
    fig.tight_layout()
    fig.savefig('LRP_barplot.png')

