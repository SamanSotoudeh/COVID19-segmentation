'''

this function contains all parameters for training the model.


batchsize:
    batch size for training the model
numepochs:
    number of epochs
input_shape:
    size of the input images and output masks
start_lr:
    starting learning rate (the default optimizer is Adam, but it can be changed)
num_folds:
    number of folds for cross-validation
threshArray:
    the segmentation map has values ranging from 0 to 1 and can be thresholded 
    at any particular value (usually, 0.50). by setting the threshArray to a
    range of numbers, the performance at different thresholds can be assessed
    (in terms of dice similarity coefficient).
verbose:
    whether to display extended information (1) or not (0)
doSave:
    whether to save the history plot for accuracy and loss metrics (1) or not (0)


MODELTYPE:
    can be selected from "UNET" or "LINKNET" without additional changes to code.
    for further selections, please refer to the following link:
    https://github.com/qubvel/segmentation_models
BACKBONE:
    please refer to the above link for full selection.
lossfunc:
    loss function used for optimization (default is binary_crossentropy, but
    can be changed to dice loss using the function provided in helpers function).
loadweights:
    can be selected from None or 'imagenet'. all backbones have pretrained weights.
    please adjust input_shape parameter accordingly, if using pretrained weights.


earlystop_monitor:
    Quantity to be monitored.
earlystop_patience_epoch:
    Number of epochs with no improvement after which training will be stopped.
reducelronplateau_monitor:
    Quantity to be monitored.
reducelronplateau_factor:
    factor by which the learning rate will be reduced.
reducelronplateau_patience:
    number of epochs with no improvement after which learning rate will be reduced.
reducelronplateau_minlr:
    lower bound on the learning rate.

checkpoint_savepath:
    string or PathLike, path to save the model file.
checkpoint_savebest:
    if save_best_only=True, it only saves when the model is considered the 
    "best" and the latest best model according to the quantity monitored will 
    not be overwritten.
checkpoint_saveweights:
    if True, then only the model's weights will be saved (model.save_weights(filepath)), 
    else the full model is saved (model.save(filepath)).
'''

import numpy as np

#%%

path = '/enter/path/here/'

# PARAMETERS        
batchsize   = 1
numepochs   = 100
input_shape = (256,256,1)
start_lr    = 1e-4
num_folds   = 5
threshArray = np.linspace(0,1,21)
verbose     = 1
doSave      = 1

MODELTYPE   = 'UNET'
BACKBONE    = 'vgg16'
lossfunc    = "binary_crossentropy"
loadweights = None

earlystop_monitor          = 'val_loss'
earlystop_patience_epoch   = 10

reducelronplateau_monitor  = 'val_loss'
reducelronplateau_factor   = 0.1
reducelronplateau_patience = 3
reducelronplateau_minlr    = 1e-7

checkpoint_savepath        = path + MODELTYPE + '_' + BACKBONE
checkpoint_savebest        = True
checkpoint_saveweights     = True

