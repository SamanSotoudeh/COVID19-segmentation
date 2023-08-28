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

import keras
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

# MODEL PARAMETERS
data_format                = params.data_format
MODELTYPE                  = params.MODELTYPE
BACKBONE                   = params.BACKBONE
imageSize                  = params.imageSize
input_shape                = params.input_shape
closing                    = params.closing
lossfunc                   = params.lossfunc
batchsize                  = params.batchsize
numepochs                  = params.numepochs
start_lr                   = params.start_lr
loadweights                = params.loadweights
num_folds                  = params.num_folds
threshArray                = params.threshArray
verbose                    = params.verbose

# CALLBACKS PARAMETERS
earlystop_patience_epoch   = params.earlystop_patience_epoch
reducelronplateau_factor   = params.reducelronplateau_factor
reducelronplateau_patience = params.reducelronplateau_patience
reducelronplateau_minlr    = params.reducelronplateau_minlr
checkpoint_savepath        = params.path + MODELTYPE + '_'+BACKBONE
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

path             = params.path

covid1Path       = path+'Final_nii_SAMPLE/E/COVID/'
normal1Path      = path+'Final_nii_SAMPLE/E/NORMAL/'
covid2Path       = path+'Final_nii_SAMPLE/T/COVID/'

# read data
Xcovid1 , ycovid1  = dataloader.covidLoader(covid1Path, imageSize, closing)
Xcovid2 , ycovid2  = dataloader.covidLoader(covid2Path, imageSize, closing)
Xnormal1, ynormal1 = dataloader.normalLoader(normal1Path)

#%% k-Fold cross-validation

covid_train, covid_test, normal_train, normal_test = helpers.get_split_indices(Xcovid1, Xnormal1, num_folds)

#%% training

for i in range(num_folds):
    
    X_train, X_valid, X_test1, y_train, y_valid, y_test1 = helpers.processDataMain(i, data_format, 
                                                                                   Xcovid1, ycovid1, 
                                                                                   Xnormal1, ynormal1,
                                                                                   covid_train, covid_test,
                                                                                   normal_train, normal_test)
    
    X_test2, y_test2 = helpers.processDataTest(data_format, Xcovid2, ycovid2, X_train)
    
    # define model
    if MODELTYPE == 'UNET':
        model = sm.Unet(BACKBONE, encoder_weights=loadweights, input_shape=(256, 256, 1))
    elif MODELTYPE == 'LINKNET':
        model = sm.Linknet(BACKBONE, encoder_weights=loadweights, input_shape=input_shape)

    model.summary()
    
    model.compile(
        optimizer=optimizers.Adam(start_lr),
        loss=lossfunc,
        metrics=[sm.metrics.iou_score, 'accuracy'],
    )
    
    callbacks = [
        EarlyStopping(patience=earlystop_patience_epoch, verbose=verbose),
        ReduceLROnPlateau(factor=reducelronplateau_factor, 
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
    
    plt.figure(figsize=(8, 8))
    plt.title("Learning curve")
    plt.plot(results.history["loss"], label="loss")
    plt.plot(results.history["val_loss"], label="val_loss")
    plt.plot(np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("log_loss")
    plt.legend()
    plt.savefig(checkpoint_savepath+'/fold_%d.jpg' %(i+1), quality=90)
    plt.show()
    
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




