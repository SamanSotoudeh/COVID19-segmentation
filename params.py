'''
this function contains all parameters for training the model
'''

import numpy as np

#%%

path = 'D:/Projects/ijr_covid19/'

# PARAMETERS        
batchsize   = 8
numepochs   = 100
imageSize   = 256
closing     = 1
input_shape = (imageSize,imageSize,1)
start_lr    = 1e-4
num_folds   = 5
threshArray = np.linspace(0,1,21)
verbose     = 1

data_format = 'channels_first'
MODELTYPE   = 'UNET'
BACKBONE    = 'vgg16'
lossfunc    = "binary_crossentropy"
loadweights = None

earlystop_patience_epoch   = 10
reducelronplateau_factor   = 0.1
reducelronplateau_patience = 3
reducelronplateau_minlr    = 1e-7
checkpoint_savepath        = path+MODELTYPE+'_'+BACKBONE
checkpoint_savebest        = True
checkpoint_saveweights     = True

