 
# standard library imports
from __future__ import absolute_import, division, print_function

# standard numerical library imports
import os
import argparse
import numpy as np
import h5py
import sys
sys.path.append('C:/Users/lucbu/Documents/Master Thesis/Data Roberto/CNN/CNN_luc/')

#from utils import ftop
#from ResCNN import ResCNN
from utils.data_utils import *
from models.networkBuilder import *

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler 
from sklearn.metrics import roc_auc_score, roc_curve

home_data = 'C:/Users/lucbu/Documents/Master Thesis/Data Roberto/CNN/CNN_luc/'
save_best_file = "C:/Users/lucbu/Documents/Master Thesis/Zhongyi/models/best_CNN.txt"

##########


def get_arguments() -> argparse.Namespace:
    """
    Set up an ArgumentParser to get the command line arguments.

    Returns:
        A Namespace object containing all the command line arguments
        for the script.
    """

    # Set up parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument('--tag',
                        default='1',
                        type=str,
                        help='tag for the output '
                             'Default: 1')
    parser.add_argument('--batch_size',
                        default=1000,
                        type=int,
                        metavar='N',
                        help='Size of the mini-batches during training. '
                             'Default: 2000.')
    parser.add_argument('--epochs',
                        default=30,
                              type=int,
                        metavar='N',
                        help='Total number of training epochs. Default: 64.')
#    parser.add_argument('--learning-rate',
#                        default=3e-4,
#                        type=float,
#                        metavar='LR',
#                        help='Initial learning rate. Default: 3e-4.')
    parser.add_argument('--resume',
                        default='False',
                        type=str,
                        help='Whether resume '
                             'Default: False')
    parser.add_argument('--tensorboard',
                        action='store_true',
                        default=True,
                        help='Use TensorBoard to log training progress? '
                             'Default: True.')
    parser.add_argument('--model_filename',
                        default=None,
                        type=str,
                        metavar='PATH',
                        help='Path to checkpoint to be used when resuming '
                             'Default: None')
    parser.add_argument('--data_filename',
                        default=None,
                        type=str,
                        metavar='PATH',
                        help='Path to data '
                             'Default: None.')
    parser.add_argument('--train',
                        default='True',
                        type=str,
                        help='Whether train or test '
                             'Default: True.')



    # Parse and return the arguments (as a Namespace object)
    arguments = parser.parse_args()
    return arguments


def load_h5(h5_filename,mode='class',unsup=False,glob=False,nevts=-1):

  global_pl = []
  f = h5py.File(h5_filename,'r')
  nevts=int(nevts)
  data = f['data']
  
  if mode == 'class':
    label = f['pid']
  elif mode == 'seg':
    label = f['label']
  else:
    print('No mode found')
  if glob:
    global_pl = f['global']
    return (data, label,global_pl)

  print("loaded {0} events".format(data.shape[1]))

  data = [data[key] for key in range(data.shape[0])]

  label = to_categorical(label, num_classes=2)


  return (data, label)


def plot_history(history, yrange):

    '''Plot loss and accuracy as a function of the epoch,
    for the training and validation datasets.
    '''
    acc = history.history['auc']
    val_acc = history.history['val_auc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Get number of epochs
    epochs = range(len(acc))

    # Plot training and validation accuracy per epoch
    plt.plot(epochs, acc)
    plt.plot(epochs, val_acc)
    plt.title('Training and validation AUC')
    plt.ylim(yrange)
    
    # Plot training and validation loss per epoch
    plt.figure()

    plt.plot(epochs, loss)
    plt.plot(epochs, val_loss)
    plt.title('Training and validation loss')
    
    plt.show()
    


"""
def model_fit(model, train_inputs, train_targets, val_inputs, val_targets, n_epoch, batch_size=32):

    hist = model.fit_generator(
        Datagen(train_inputs, train_targets, batch_size, is_train=True),
        steps_per_epoch = len(train_inputs) // batch_size,
        epochs = n_epoch,
        validation_data=Datagen(val_inputs, val_targets, batch_size),
        validation_steps = len(val_inputs) // batch_size,
        callbacks = [lr_schedule, macroF1(model, val_inputs, val_targets)],
        shuffle = False,
        verbose = 1
        )
    return hist
"""

def load_data():

 #load data
 X_train, Y_train = load_h5(home_data + 'data/' + 'train_cnn1d_splitted.h5')

 X_val, Y_val = load_h5(home_data + 'data/' + 'validation_cnn1d_splitted.h5')

 X_test, Y_test = load_h5(home_data + 'data/' + 'test_cnn1d_splitted.h5')

 #Normalization
 

 return X_train, X_val, X_test, Y_train, Y_val, Y_test


def train(X_train, X_val, Y_train, Y_val, model, args):
 Nexpert = 0
 Ntotal = len(X_train) + Nexpert
 input_size = X_train[0].shape[1]

 #schedulers

 n_epoch = args.epochs
 batch_size = args.batch_size 

 tl_checkpoint = ModelCheckpoint(filepath= home_data + 'weights/weights.best.hdf5',
                                    monitor='val_auc',
                                    mode='max',
                                    save_best_only=True,
                                    verbose=1)

 early_stop = EarlyStopping(monitor='val_auc',
                             patience=5,
                             restore_best_weights=True,
                             mode='max')

 reduce_lr = ReduceLROnPlateau(monitor='val_auc', 
                                factor=0.5, 
                                patience=3, 
                                verbose=1,
                                mode='max',
                                min_lr=0.000001)

 tb = TensorBoard(log_dir='./tensorboard')

 # train model
 hist = model.fit(X_train, Y_train,
            epochs=n_epoch,
            batch_size=batch_size,
            validation_data=(X_val, Y_val),
            callbacks=[tl_checkpoint, reduce_lr, tb],
            verbose=1)

 plot_history(hist,(0.75,0.85))

def test(X_test, Y_test, model, args):
 model.load_weights('C:/Users/lucbu/Documents/Master Thesis/Data Roberto/CNN/CNN_luc/weights/weights.best.hdf5')
 
 print(model.evaluate(X_test, Y_test))

 batch_size = args.batch_size

 preds = model.predict(X_test, batch_size)

 print(np.shape(preds))
 print(np.shape(X_test))
 print(np.shape(Y_test))

 AUC = roc_auc_score(Y_test[:,1], preds[:,1])
 
 fpr, tpr, threshold = roc_curve(Y_test[:,1], preds[:,1])

 print('AUC', AUC)
 
 np.savetxt('C:/Users/lucbu/Documents/Master Thesis/Results/scores/CNN.csv', preds[:,1])
 
 plt.plot(fpr, tpr)
 plt.xlabel('FPR')
 plt.ylabel('TPR')
 plt.title('ROC curve')
 plt.show()
 
 labels0 = []
 labels1 = []
 
 for i in range(len(Y_test[:,1])):
     if Y_test[i,1] == 1:
         labels1.append(i)
     else:
         labels0.append(i)

 plt.hist(preds[labels1,1], bins=20, alpha=0.7, label='signal')
 plt.hist(preds[labels0,1], bins=20, alpha=0.7, label='background')
 plt.legend(loc='best')
 plt.xlabel('Network ouput')
 plt.ylabel('Frequency')
 plt.show()
 

 
 #import h5py
 
 #with h5py.File('C:/Users/lucbu/Documents/Master Thesis/Data Roberto/CNN/CNN_luc/weights/FPR.h5', "w") as hf:
 #     hf.create_dataset("fpr", data=fpr)
 #     hf.create_dataset("tpr", data=tpr)
 #     hf.create_dataset("threshold", data=threshold)
 
 #with h5py.File('C:/Users/lucbu/Documents/Master Thesis/Data Roberto/CNN/CNN_luc/weights/output.h5', "w") as hf:
 #     hf.create_dataset("signal", data=preds[labels1,1])
 #     hf.create_dataset("background", data=preds[labels0,1])
 
 


# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------
    
    # Read in command line arguments
    args = get_arguments()
  
    X_train, X_val, X_test, Y_train, Y_val, Y_test = load_data()

    #Number of high level features 
    Nexpert = 0
    Ntotal = len(X_train) + Nexpert
    input_size = X_train[0].shape[1]

    model = build_CNN_1D(Ntotal, Nexpert, input_size)
    print(model.summary())


    if args.train == 'True':
      
      train(X_train, X_val, Y_train, Y_val, model, args)

      print('')
      print('TRAINED MODEL')
      print('')
      
      test(X_test, Y_test, model,  args)
     

    else:

      print('')
      print('STARTING PREDICTIONS')
      print('') 

      test(X_test, Y_test, model,  args)

      print('')
      print('PREDICTIONS ON TEST DATA DONE')
      print('')
