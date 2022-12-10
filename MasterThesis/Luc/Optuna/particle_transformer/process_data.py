import torch
import numpy as np
import tqdm 
import pandas as pd
import h5py
import math

def convert_data(data):
    num_objects = 12

    n_data = data.shape[0]

    # aux data
    data_aux = data[:,:8]
    
    # jet, b-jet, e-, e+, mu-, mu+
    # Set up a set of corresponding categories
    data_ids = torch.zeros((n_data, 6, num_objects), dtype=torch.bool)
    for i in range(0,4):
        data_ids[:,0,i] = True
    for i in range(4,8):
        data_ids[:,1,i] = True
    data_ids[:,2,8] = True
    data_ids[:,3,9] = True
    data_ids[:,4,10] = True
    data_ids[:,5,11] = True

    # Momenta in  E, pt, eta, phi
    data_momenta = data[:,8:].reshape(n_data, num_objects, 4)

    # Momenta in E, px, py, pz
    data_four_vec = torch.zeros_like(data_momenta)
    data_four_vec[:,:,0] = torch.exp(data_momenta[:,:,1])*torch.cos(data_momenta[:,:,3])
    data_four_vec[:,:,1] = torch.exp(data_momenta[:,:,1])*torch.sin(data_momenta[:,:,3])
    data_four_vec[:,:,2] = torch.exp(data_momenta[:,:,1])*torch.sinh(data_momenta[:,:,2])
    data_four_vec[:,:,3] = torch.exp(data_momenta[:,:,0])

    # Set back to zero
    data_four_vec[data_momenta[:,:,0] == 0.] = 0.

    # Permute
    data_momenta = torch.permute(data_momenta, (0,2,1))
    data_four_vec = torch.permute(data_four_vec, (0,2,1))
    
    # Tokens are concat of momenta and ids
    data_tokens = torch.cat((data_momenta, data_ids), dim=1)
    
    # Generate padding mask
    data_mask = (data_momenta[:,0,:] != 0.).unsqueeze(1)
    
    return data_aux, data_tokens, data_four_vec, data_mask

def get_training_dataloaders(n_batches, data_loc):
    # Load all the data
    hf = h5py.File(data_loc, 'r')

    data_train  = torch.tensor(np.array(hf.get('X_train')), dtype=torch.float32)
    data_val    = torch.tensor(np.array(hf.get('X_val')), dtype=torch.float32)
    data_test   = torch.tensor(np.array(hf.get('X_test')), dtype=torch.float32)
    labels_train = torch.tensor(hf.get('Y_train'), dtype=torch.long)
    labels_val   = torch.tensor(hf.get('y_val'), dtype=torch.long)
    labels_test  = torch.tensor(hf.get('y_test'), dtype=torch.long)

    # Combine the data
    data_total = torch.cat((data_train, data_val, data_test), dim=0)

    # Normalize
    '''
    for i in range(6, data_total.shape[1]):
        mask = data_total[:,i] != 0.
        masked = data_total[mask,i]

        mean = torch.mean(masked)
        std = torch.std(masked)

        data_train[data_train[:,i] != 0., i] = (data_train[data_train[:,i] != 0., i] - mean)/std
        data_val[data_val[:,i] != 0., i]     = (data_val[data_val[:,i] != 0., i]     - mean)/std
        data_test[data_test[:,i] != 0., i]   = (data_test[data_test[:,i] != 0., i]   - mean)/std
    '''
    data_aux_train, data_tokens_train, data_momenta_train, data_mask_train = convert_data(data_train)
    data_aux_val,   data_tokens_val,   data_momenta_val,   data_mask_val   = convert_data(data_val)
    data_aux_test,  data_tokens_test,  data_momenta_test,  data_mask_test  = convert_data(data_test)

    # Combine
    data_train_combined = torch.utils.data.TensorDataset(data_aux_train, data_tokens_train, data_momenta_train, data_mask_train, labels_train)
    data_val_combined   = torch.utils.data.TensorDataset(data_aux_val,   data_tokens_val,   data_momenta_val,   data_mask_val,   labels_val)
    data_test_combined  = torch.utils.data.TensorDataset(data_aux_test,  data_tokens_test,  data_momenta_test,  data_mask_test,  labels_test)
    
    # Make loader
    batch_size = math.ceil(data_train.shape[0]/n_batches)

    train_loader = torch.utils.data.DataLoader(dataset=data_train_combined, batch_size=batch_size, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(dataset=data_val_combined,   batch_size=batch_size)
    test_loader  = torch.utils.data.DataLoader(dataset=data_test_combined,  batch_size=batch_size)

    return train_loader, val_loader, test_loader