#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import numpy as np
import awkward
import uproot_methods
import h5py as h5

import logging
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(levelname)s: %(message)s')

splitted = True

def data_split(*args, **kwargs):

    """A function to split a dataset into train, test, and optionally 
    validation datasets.

    **Arguments**

    - ***args** : arbitrary _numpy.ndarray_ datasets
        - An arbitrary number of datasets, each required to have
        the same number of elements, as numpy arrays.
    - **train** : {_int_, _float_}
        - If a float, the fraction of elements to include in the training
        set. If an integer, the number of elements to include in the
        training set. The value `-1` is special and means include the
        remaining part of the dataset in the training dataset after
        the test and (optionally) val parts have been removed
    - **val** : {_int_, _float_}
        - If a float, the fraction of elements to include in the validation
        set. If an integer, the number of elements to include in the
        validation set. The value `0` is special and means do not form
        a validation set.
    - **test** : {_int_, _float_}
        - If a float, the fraction of elements to include in the test
        set. If an integer, the number of elements to include in the
        test set.
    - **shuffle** : _bool_
        - A flag to control whether the dataset is shuffled prior to
        being split into parts.

    **Returns**

    - _list_
        - A list of the split datasets in train, [val], test order. If 
        datasets `X`, `Y`, and `Z` were given as `args` (and assuming a
        non-zero `val`), then [`X_train`, `X_val`, `X_test`, `Y_train`, 
        `Y_val`, `Y_test`, `Z_train`, `Z_val`, `Z_test`] will be returned.
    """

    # handle valid kwargs
    train, val, test = kwargs.pop('train', -1), kwargs.pop('val', 0.0), kwargs.pop('test', 0.1)
    shuffle = kwargs.pop('shuffle', True)
    if len(kwargs):
        raise TypeError('following kwargs are invalid: {}'.format(kwargs))

    # validity checks
    if len(args) == 0: 
        raise RuntimeError('Need to pass at least one argument to data_split')

    # check for consistent length
    n_samples = len(args[0])
    for arg in args[1:]: 
        assert len(arg) == n_samples, 'args to data_split have different length'

    # determine numbers
    num_val = int(n_samples*val) if val<=1 else val
    num_test = int(n_samples*test) if test <=1 else test
    num_train = n_samples - num_val - num_test if train==-1 else (int(n_samples*train) if train<=1 else train)
    assert num_train >= 0, 'bad parameters: negative num_train'
    assert num_train + num_val + num_test <= n_samples, 'too few samples for requested data split'
    
    # calculate masks 
    perm = np.random.permutation(n_samples) if shuffle else np.arange(n_samples)
    train_mask = perm[:num_train]
    val_mask = perm[-num_val:]
    test_mask = perm[num_train:num_train+num_test]

    # apply masks
    masks = [train_mask, val_mask, test_mask] if num_val > 0 else [train_mask, test_mask]

    # return list of new datasets
    return [arg[mask] for arg in args for mask in masks]

def _transform(dataframe, start=0, stop=-1, jet_size=0.8):
    from collections import OrderedDict
    v = OrderedDict()

    df = dataframe.iloc[start:stop]
    def _col_list(prefix, max_particles=200):
        return ['%s_%d'%(prefix,i) for i in range(max_particles)]
    
    _px = df[_col_list('PX')].values
    _py = df[_col_list('PY')].values
    _pz = df[_col_list('PZ')].values
    _e = df[_col_list('E')].values
    
    mask = _e>0
    n_particles = np.sum(mask, axis=1)

    px = awkward.JaggedArray.fromcounts(n_particles, _px[mask])
    py = awkward.JaggedArray.fromcounts(n_particles, _py[mask])
    pz = awkward.JaggedArray.fromcounts(n_particles, _pz[mask])
    energy = awkward.JaggedArray.fromcounts(n_particles, _e[mask])

    p4 = uproot_methods.TLorentzVectorArray.from_cartesian(px, py, pz, energy)
    pt = p4.pt

    jet_p4 = p4.sum()

    # outputs
    _label = df['is_signal_new'].values
    v['label'] = np.stack((_label, 1-_label), axis=-1)
    v['train_val_test'] = df['ttv'].values
    
    v['jet_pt'] = jet_p4.pt
    v['jet_eta'] = jet_p4.eta
    v['jet_phi'] = jet_p4.phi
    v['jet_mass'] = jet_p4.mass
    v['n_parts'] = n_particles

    v['part_px'] = px
    v['part_py'] = py
    v['part_pz'] = pz
    v['part_energy'] = energy

    v['part_pt_log'] = np.log(pt)
    v['part_ptrel'] = pt/v['jet_pt']
    v['part_logptrel'] = np.log(v['part_ptrel'])

    v['part_e_log'] = np.log(energy)
    v['part_erel'] = energy/jet_p4.energy
    v['part_logerel'] = np.log(v['part_erel'])

    v['part_raw_etarel'] = (p4.eta - v['jet_eta'])
    _jet_etasign = np.sign(v['jet_eta'])
    _jet_etasign[_jet_etasign==0] = 1
    v['part_etarel'] = v['part_raw_etarel'] * _jet_etasign

    v['part_phirel'] = p4.delta_phi(jet_p4)
    v['part_deltaR'] = np.hypot(v['part_etarel'], v['part_phirel'])

    def _make_image(var_img, rec, n_pixels = 64, img_ranges = [[-0.8, 0.8], [-0.8, 0.8]]):
        wgt = rec[var_img]
        x = rec['part_etarel']
        y = rec['part_phirel']
        img = np.zeros(shape=(len(wgt), n_pixels, n_pixels))
        for i in range(len(wgt)):
            hist2d, xedges, yedges = np.histogram2d(x[i], y[i], bins=[n_pixels, n_pixels], range=img_ranges, weights=wgt[i])
            img[i] = hist2d
        return img

#     v['img'] = _make_image('part_ptrel', v)

    return v

def convert(clas, reg, y, destdir, basename):

   from collections import OrderedDict
   v = OrderedDict()

   #reg = reg[:,]
   #clas = clas[:,]

   e_pos = list(range(2,reg.shape[1],4))
   pt_pos = list(range(3,reg.shape[1],4))
   phi_pos = list(range(4,reg.shape[1],4))
   eta_pos = list(range(5,reg.shape[1],4))

   met = reg[:,0] #np.exp(reg[:,0])
   met_phi = reg[:,1]
   met_eta = np.zeros(reg.shape[0])
   met_clas = 9.*np.ones(reg.shape[0])

   _pt = reg[:,pt_pos]
   _eta = reg[:,eta_pos]
   _phi = reg[:,phi_pos]
   _e = reg[:,e_pos]
   _clas = clas[:]  

   """
   #normalization event by event
   for pt in _pt:
    mask = pt > 0
    pt[mask] = np.exp(pt[mask])

   _pt_sum = met[:] + np.sum(_pt, axis = 1)

   met = met/_pt_sum
   indx = 0
   for pt in _pt:
     pt /= _pt_sum[indx]
     indx += 1
   """
   _pt = np.concatenate((met.reshape(-1,1), _pt), axis=1)
   _eta = np.concatenate((met_eta.reshape(-1,1), _eta), axis=1)
   _phi = np.concatenate((met_phi.reshape(-1,1), _phi), axis=1)
   _e = np.concatenate((met.reshape(-1,1), _e), axis=1)
   _clas = np.concatenate((met_clas.reshape(-1,1), _clas), axis=1)
   
   
   #Normalize
   #_pt = _pt/np.linalg.norm(_pt)
   #_eta = _eta/np.linalg.norm(_eta)
   #_phi = _phi/np.linalg.norm(_phi)
   #_e = _e/np.linalg.norm(_e)
   
   
   NFEAT=11

   """
   points = np.zeros((_pt.shape[0], _pt.shape[1], NFEAT))

   points[:,:,0] = _eta
   points[:,:,1] = _phi
   points[:,:,2] = _pt
   points[:,:,3] = _e
   points[:,:,4] = np.abs(_clas[:])==1
   points[:,:,5] = np.abs(_clas[:])==2
   points[:,:,6] = np.abs(_clas[:])==3
   points[:,:,7] = np.abs(_clas[:])==4
   points[:,:,8] = np.abs(_clas[:])==5
   points[:,:,9] = np.abs(_clas[:])==6
   points[:,:,10] = np.abs(_clas[:])==7
   """
   points = np.zeros((NFEAT, _pt.shape[0], _pt.shape[1]))
   
   points[0,:,:] = _eta
   points[1,:,:] = _phi
   points[2,:,:] = _pt
   points[3,:,:] = _e
   points[4,:,:] = np.abs(_clas[:])==1
   points[5,:,:] = np.abs(_clas[:])==2
   points[6,:,:] = np.abs(_clas[:])==3
   points[7,:,:] = np.abs(_clas[:])==4
   points[8,:,:] = np.abs(_clas[:])==5
   points[9,:,:] = np.abs(_clas[:])==6
   points[10,:,:] = np.abs(_clas[:])==7
   
   

   points = np.expand_dims(points, axis=3)
   #print(points[0,0]) 
   npid = y

   output_name = os.path.join(destdir, basename)

   with h5.File(output_name, "w") as fh5:
        dset = fh5.create_dataset("data", data=points)
        dset = fh5.create_dataset("pid", data=npid)


def load(fname, num_data=50000, cache_dir='~/.energyflow'):

    print ('Unpacking file', fname) 
    
    hf = h5.File(fname, 'r')

    y = hf.get('type')
    nobj = hf.get('nobj')
    reg = hf.get('reg')
    clas = hf.get('clas')

    y = np.array(y, dtype=np.float32)
    nobj = np.array(nobj)
    reg = np.array(reg)
    clas = np.array(clas, dtype=np.float32)

    #Filter 
    """
    ttbarWW : 1
    ttbarZ_lep : 2
    ttbarW_lep : 3
    ttbarHiggs_lep : 4
    4top_1jet : 5
    """

    indx=(y==5)

    ftop_nobj = nobj[indx]
    ftop_reg = reg[indx]
    ftop_clas = clas[indx]

    indx=(y==3)

    ttbarW_nobj = nobj[indx]
    ttbarW_reg = reg[indx]
    ttbarW_clas = clas[indx]

    indx=(y==2)

    ttbarZ_nobj = nobj[indx]
    ttbarZ_reg = reg[indx]
    ttbarZ_clas = clas[indx]

    indx=(y==1)

    ttbarWW_nobj = nobj[indx]
    ttbarWW_reg = reg[indx]
    ttbarWW_clas = clas[indx]

    indx=(y==4)

    ttbarH_nobj = nobj[indx]
    ttbarH_reg = reg[indx]
    ttbarH_clas = clas[indx]

    ftop_type = np.ones(ftop_nobj.shape[0])
    ttbarW_type = np.zeros(ttbarW_nobj.shape[0])
    ttbarZ_type = np.zeros(ttbarZ_nobj.shape[0])
    ttbarH_type = np.zeros(ttbarH_nobj.shape[0])
    ttbarWW_type = np.zeros(ttbarWW_nobj.shape[0])


    #Choose max. 50k events per process excepting 4tops which 
    #is then 200k to get a balanced dataset
    max_num_events = num_data #50000
    if ftop_nobj.shape[0] > 4.*max_num_events:
     ftop_nobj = ftop_nobj[:4*max_num_events]
     ftop_reg = ftop_reg[:4*max_num_events]
     ftop_type = np.ones(4*max_num_events)
     ftop_clas = ftop_clas[:4*max_num_events]

    if ttbarW_nobj.shape[0] > max_num_events:
     ttbarW_nobj = ttbarW_nobj[:max_num_events]
     ttbarW_reg = ttbarW_reg[:max_num_events]
     ttbarW_type = np.zeros(max_num_events)
     ttbarW_clas = ttbarW_clas[:max_num_events]

    if ttbarZ_nobj.shape[0] > max_num_events:
     ttbarZ_nobj = ttbarZ_nobj[:max_num_events]
     ttbarZ_reg = ttbarZ_reg[:max_num_events]
     ttbarZ_type = np.zeros(max_num_events)
     ttbarZ_clas = ttbarZ_clas[:max_num_events]

    if ttbarH_nobj.shape[0] > max_num_events:
     ttbarH_nobj = ttbarH_nobj[:max_num_events]
     ttbarH_reg = ttbarH_reg[:max_num_events]
     ttbarH_type = np.zeros(max_num_events)
     ttbarH_clas = ttbarH_clas[:max_num_events]

    if ttbarWW_nobj.shape[0] > max_num_events:
     ttbarWW_nobj = ttbarWW_nobj[:max_num_events]
     ttbarWW_reg = ttbarWW_reg[:max_num_events]
     ttbarWW_type = np.zeros(max_num_events)
     ttbarWW_clas = ttbarWW_clas[:max_num_events]

    #print(ftop_nobj.shape,  ttbarW_nobj.shape, ttbarZ_nobj.shape, ttbarH_nobj.shape, ttbarWW_nobj.shape, ftop_clas.shape)

    #Now I have to concatenate and suffle(perm)
    _nobj = np.concatenate((ftop_nobj, ttbarW_nobj, ttbarZ_nobj, ttbarH_nobj, ttbarWW_nobj))
    _reg = np.vstack((ftop_reg, ttbarW_reg, ttbarZ_reg, ttbarH_reg, ttbarWW_reg))
    _clas = np.vstack((ftop_clas, ttbarW_clas, ttbarZ_clas, ttbarH_clas, ttbarWW_clas))
    _y = np.concatenate((ftop_type, ttbarW_type, ttbarZ_type, ttbarH_type, ttbarWW_type)) 

    #randomize
    perm = np.random.permutation(_nobj.shape[0])

    _nobj = np.reshape(_nobj, (-1,1))

    #print(_reg.shape, _clas.shape,np.reshape(_nobj, (-1,1)).shape)
    nobj = _nobj[perm] 
    reg = _reg[perm] 
    y = _y[perm] 
    clas = _clas[perm]

    #print(type(nobj), type(reg), nobj[0], reg[0])

    #X = np.concatenate((nobj, clas, reg), axis = 1) 

    #if num_data > -1:
    #    X, y = X[:num_data], y[:num_data]

    return clas, reg, y

def load_splitted(fname):

    print ('Unpacking file', fname) 

    hf = h5.File(fname, 'r') 

    X_train = np.array(hf['X_train']) 
    X_val = np.array(hf['X_val'])
    X_test = np.array(hf['X_test'])

    y_train = np.array(hf['Y_train'])
    y_val = np.array(hf['y_val'])
    y_test = np.array(hf['y_test'])

    return X_train, X_val, X_test, y_train, y_val, y_test

srcDir = 'C:/Users/lucbu/Documents/Master Thesis/Data Roberto/CNN/CNN_luc/data/'
destDir = 'C:/Users/lucbu/Documents/Master Thesis/Data Roberto/CNN/CNN_luc/data/'

if splitted:

 # load data
 X_train, X_val, X_test, y_train, y_val, y_test = load_splitted(srcDir + '4top_400k_orig_splitted.h5') #, num_data=num_data)
 
 
 clas_train = X_train[:, :19]
 clas_val = X_val[:, :19]
 clas_test = X_test[:, :19]
 reg_train = X_train[:, 19:]
 reg_val = X_val[:, 19:]
 reg_test = X_test[:, 19:]

 #print(X_train.shape, X_train[0], clas_train[0], reg_train[0])

 #sys.exit()
 #normalize

else:

 clas, reg, y = load(os.path.join(srcDir, '4top_400k_orig_splitted.h5'))

 val_frac = 0.1
 test_frac = 0.1

 # convert training file
 #convert(clas, reg, y, destdir=destDir, basename='4top_cnn1d.h5')

 #split
 (clas_train, clas_val, clas_test, reg_train, reg_val, reg_test, y_train, y_val, y_test) = data_split(clas, reg, y, val=val_frac, test=test_frac)


# convert training file
convert(clas_train, reg_train, y_train, destdir=destDir, basename='train_cnn1d_splitted.h5')

# convert validation file
convert(clas_val, reg_val, y_val, destdir=destDir, basename='validation_cnn1d_splitted.h5')

# convert testing file
convert(clas_test, reg_test, y_test, destdir=destDir, basename='test_cnn1d_splitted.h5')






