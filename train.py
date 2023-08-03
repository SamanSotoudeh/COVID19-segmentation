'''

Please ensure to cite the relevant article when using this code in your work.

Sotoudeh-Paima, S., Hasanzadeh, N., Bashirgonbadi, A., Aref, A., Naghibi, M., 
Zoorpaikar, M., ... & Soltanian-Zadeh, H. (2022). A Multi-centric Evaluation 
of Deep Learning Models for Segmentation of COVID-19 Lung Lesions on Chest CT 
Scans. Iranian Journal of Radiology, 19(4).

Author: Saman Sotoudeh-Paima
Date: August 2nd, 2023

'''

#%% importing packages

import os
import params
import helpers
import dataloader
import numpy as np
import keras.backend as K
import segmentation_models as sm
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import optimizers

#%%

path                       = params.path
savePath                   = path + 'savedFiles/'

# MODEL PARAMETERS
MODELTYPE                  = params.MODELTYPE
BACKBONE                   = params.BACKBONE
input_shape                = params.input_shape
lossfunc                   = params.lossfunc
batchsize                  = params.batchsize
numepochs                  = params.numepochs
start_lr                   = params.start_lr
loadweights                = params.loadweights
num_folds                  = params.num_folds
threshArray                = params.threshArray
doSave                     = params.doSave
verbose                    = params.verbose

# CALLBACKS PARAMETERS
# early stopping
earlystop_monitor          = params.earlystop_monitor
earlystop_patience_epoch   = params.earlystop_patience_epoch
# reduce learning rate on plateau
reducelronplateau_monitor  = params.reducelronplateau_monitor
reducelronplateau_factor   = params.reducelronplateau_factor
reducelronplateau_patience = params.reducelronplateau_patience
reducelronplateau_minlr    = params.reducelronplateau_minlr
# model checkpoint
checkpoint_savepath        = savePath + MODELTYPE + '_' + BACKBONE
checkpoint_savebest        = params.checkpoint_savebest
checkpoint_saveweights     = params.checkpoint_saveweights

#%% initialization

threshLen          = len(threshArray)
 
scores_load_val    = np.zeros(num_folds)
scores_load_test1  = np.zeros(num_folds)
scores_load_test2  = np.zeros(num_folds)

dices_valid        = np.zeros([num_folds, threshLen])
dices_test1        = np.zeros([num_folds, threshLen])
dices_test2        = np.zeros([num_folds, threshLen])

#%% dataloading

# directories of datasets 1 (containing COVID-19 and normal cases) and 2 (containing only COVID-19 cases)
ctPath_covid1    = path + 'Data/SAMPLE/Naghibi/COVID_CT/'
labelPath_covid1 = path + 'Data/SAMPLE/Naghibi/COVID_LABEL/'
ctPath_normal1   = path + 'Data/SAMPLE/Naghibi/NORMAL_CT/'

ctPath_covid2    = path + 'Data/SAMPLE/Zorpeykar/COVID_CT/'
labelPath_covid2 = path + 'Data/SAMPLE/Zorpeykar/COVID_LABEL/'

# read data
Xcovid1 , ycovid1  = dataloader.covidLoader(ctPath_covid1, labelPath_covid1)
Xnormal1, ynormal1 = dataloader.normalLoader(ctPath_normal1)
Xcovid2 , ycovid2  = dataloader.covidLoader(ctPath_covid2, labelPath_covid2)

#%% k-Fold cross-validation

covid_train, covid_test, normal_train, normal_test = helpers.get_split_indices(Xcovid1, Xnormal1, num_folds)

#%% training

os.makedirs(savePath, exist_ok=True)
os.makedirs(savePath+MODELTYPE+'_'+BACKBONE, exist_ok=True)

for i in range(num_folds):
    
    X_train, X_valid, X_test1, y_train, y_valid, y_test1 = helpers.processDataMain(i, Xcovid1, ycovid1, 
                                                                              Xnormal1, ynormal1, 
                                                                              covid_train, covid_test,
                                                                              normal_train, normal_test)
    
    X_test2, y_test2 = helpers.processDataTest(Xcovid2, ycovid2, X_train)
    
    # define model
    if MODELTYPE == 'UNET':
        model = sm.Unet(BACKBONE, encoder_weights=loadweights, input_shape=input_shape)
    elif MODELTYPE == 'LINKNET':
        model = sm.Linknet(BACKBONE, encoder_weights=loadweights, input_shape=input_shape)

    model.summary()
    
    model.compile(
        optimizer=optimizers.Adam(start_lr),
        loss=lossfunc,
        metrics=[sm.metrics.iou_score, 'accuracy'],
    )
    
    callbacks = [
        EarlyStopping(monitor=earlystop_monitor,
                      patience=earlystop_patience_epoch, 
                      verbose=verbose),
        ReduceLROnPlateau(monitor=reducelronplateau_monitor,
                          factor=reducelronplateau_factor, 
                          patience=reducelronplateau_patience,
                          min_lr=reducelronplateau_minlr, 
                          verbose=verbose),
        ModelCheckpoint(checkpoint_savepath+'/fold%d.h5' %(i+1), verbose=verbose, 
                        save_best_only=checkpoint_savebest, 
                        save_weights_only=checkpoint_saveweights)]
    
    # fit model
    # if you use data generator use model.fit_generator(...) instead of model.fit(...)
    # more about `fit_generator` here: https://keras.io/models/sequential/#fit_generator
    
    results = model.fit(x=X_train, y=y_train, batch_size=batchsize, epochs=numepochs,
                        validation_data=(X_valid, y_valid), callbacks=callbacks)

    # Visualization
    helpers.historyPlot(results, checkpoint_savepath, i, doSave)
    
    # load the best model
    model.load_weights(checkpoint_savepath+'/fold%d.h5' %(i+1))

    # Evaluate on validation set
    scores_load_val[i]   = model.evaluate(X_valid, y_valid, batch_size=batchsize, verbose=verbose)[0]
    # Evaluate on test set
    scores_load_test1[i] = model.evaluate(X_test1, y_test1, batch_size=batchsize, verbose=verbose)[0]
    # Evaluate on zorpeykar set
    scores_load_test2[i] = model.evaluate(X_test2, y_test2, batch_size=batchsize, verbose=verbose)[0]

    # Predict on validation and test and zorpeykar
    preds_val   = model.predict(X_valid, batch_size=batchsize, verbose=verbose)
    preds_test1 = model.predict(X_test1, batch_size=batchsize, verbose=verbose)
    preds_test2 = model.predict(X_test2, batch_size=batchsize, verbose=verbose)
    
    for k in range(threshLen):
        
        preds_valid_t = (preds_val > threshArray[k]).astype(np.uint8)
        preds_valid_t = preds_valid_t.astype(np.float64)
        
        preds_test1_t = (preds_test1 > threshArray[k]).astype(np.uint8)
        preds_test1_t = preds_test1_t.astype(np.float64)
        
        preds_test2_t = (preds_test2 > threshArray[k]).astype(np.uint8)
        preds_test2_t = preds_test2_t.astype(np.float64)
        
        _, dices_valid[i,k]  = helpers.dice_coef_perslice(y_valid, preds_valid_t) 
        _, dices_test1[i,k]  = helpers.dice_coef_perslice(y_test1, preds_test1_t) 
        _, dices_test2[i,k]  = helpers.dice_coef_perslice(y_test2, preds_test2_t)        

        print("(VALIDATION) Threshold: %.2f, Dice: %.4f" %(threshArray[k], dices_valid[i,k]))
        print("(TEST 1)     Threshold: %.2f, Dice: %.4f" %(threshArray[k], dices_test1[i,k]))
        print("(TEST 2)     Threshold: %.2f, Dice: %.4f" %(threshArray[k], dices_test2[i,k]))

    K.clear_session()




