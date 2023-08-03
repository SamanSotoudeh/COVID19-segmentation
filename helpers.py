'''
This code consists of helper functions.
'''

#%% importing packages

import numpy as np
import keras.backend as K
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#%%

''' 
Calculates dice similarity coefficient and dice loss.
'''

def dice_coef(y_true, y_pred, smooth=1):
    intersection  = K.sum(y_true * y_pred, axis=[1,2,3])
    union         = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

#%%

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

#%%

def dice_coef_numpy(y_true, y_pred, smooth=1e-7):
        
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    union        = np.sum(y_true) + np.sum(y_pred)
    intersection = np.sum(y_true*y_pred)
    
    dice  = (2*intersection + smooth) / (union + smooth)
    return dice

#%%

def dice_coef_perslice(y_true, y_predt, smooth=1e-7):

    dice = np.zeros(len(y_true))

    for i in range(len(y_true)):
        
        y_true_slice  = y_true[i,:,:,:]
        y_predt_slice = y_predt[i,:,:,:]
        y_true_slice  = y_true_slice.flatten()
        y_predt_slice = y_predt_slice.flatten()
        
        union         = np.sum(y_true_slice) + np.sum(y_predt_slice)
        intersection  = np.sum(y_true_slice*y_predt_slice)
        
        dice[i]  = (2*intersection + smooth) / (union + smooth)
        
    return dice, np.mean(dice)

#%%

def get_split_indices(Xcovid1, Xnormal1, num_folds):

    covid_train  = []
    covid_test   = []
    normal_train = []
    normal_test  = []
    
    kfold = KFold(n_splits = num_folds)
    
    cov_linspace    = np.linspace(0, len(Xcovid1)-1, len(Xcovid1))
    normal_linspace = np.linspace(0, len(Xnormal1)-1, len(Xnormal1))
    
    for train, test in kfold.split(cov_linspace):
        covid_train.append(train)
        covid_test.append(test)
        
    for train, test in kfold.split(normal_linspace):
        normal_train.append(train)
        normal_test.append(test)
        
    return covid_train, covid_test, normal_train, normal_test

#%%

def processDataMain(i, Xcovid1, ycovid1, Xnormal1, ynormal1, covid_train, covid_test, normal_train, normal_test):

    X_train_covid = list(Xcovid1[i] for i in covid_train[i])
    y_train_covid = list(ycovid1[i] for i in covid_train[i])
    
    X_test_covid  = list(Xcovid1[i] for i in covid_test[i])
    y_test_covid  = list(ycovid1[i] for i in covid_test[i])
    
    X_train_covid, X_valid_covid, y_train_covid, y_valid_covid = train_test_split(
            X_train_covid, y_train_covid, test_size=0.25)
    
    X_train_normal = list(Xnormal1[i] for i in normal_train[i])
    y_train_normal = list(ynormal1[i] for i in normal_train[i])
    
    X_test_normal  = list(Xnormal1[i] for i in normal_test[i])
    y_test_normal  = list(ynormal1[i] for i in normal_test[i])    
    
    X_train_normal, X_valid_normal, y_train_normal, y_valid_normal = train_test_split(
            X_train_normal, y_train_normal, test_size=0.25)

    X_train = X_train_covid + X_train_normal
    y_train = y_train_covid + y_train_normal
    
    X_valid = X_valid_covid + X_valid_normal
    y_valid = y_valid_covid + y_valid_normal
    
    X_test  = X_test_covid +  X_test_normal
    y_test  = y_test_covid +  y_test_normal
    
    # Stack X
    X_train = np.vstack(X_train)
    X_valid = np.vstack(X_valid)
    X_test1 = np.vstack(X_test)
    
    # Stack Y
    y_train = np.vstack(y_train)
    y_valid = np.vstack(y_valid)
    y_test1 = np.vstack(y_test)

    # Expand Y Dimensions
    y_train = np.expand_dims(y_train, axis =3)
    y_valid = np.expand_dims(y_valid, axis =3)
    y_test1 = np.expand_dims(y_test1, axis =3)
    
    # Binarize Y
    y_train[y_train > 0] = 1
    y_valid[y_valid > 0] = 1
    y_test1[y_test1 > 0] = 1
    
    # PREPROCESSING
    scaler          = MinMaxScaler()
    X_train_scaled  = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_valid_scaled  = scaler.transform(X_valid.reshape(-1, X_valid.shape[-1])).reshape(X_valid.shape)
    X_test1_scaled  = scaler.transform(X_test1.reshape(-1, X_test1.shape[-1])).reshape(X_test1.shape)

    # SHUFFLE            
    train_num = np.random.rand(X_train_scaled.shape[0]).argsort()        
    valid_num = np.random.rand(X_valid_scaled.shape[0]).argsort()        
    test1_num = np.random.rand(X_test1_scaled.shape[0]).argsort()        

    X_train = np.take(X_train_scaled, train_num, axis=0, out=X_train_scaled)
    X_valid = np.take(X_valid_scaled, valid_num, axis=0, out=X_valid_scaled)
    X_test1 = np.take(X_test1_scaled, test1_num , axis=0, out=X_test1_scaled)
    
    y_train = np.take(y_train, train_num, axis=0, out=y_train)
    y_valid = np.take(y_valid, valid_num, axis=0, out=y_valid)
    y_test1 = np.take(y_test1, test1_num, axis=0, out=y_test1)

    return X_train, X_valid, X_test1, y_train, y_valid, y_test1

#%%

def processDataTest(Xcovid2, ycovid2, X_train):

    X_test2 = np.vstack(Xcovid2)
    y_test2 = np.vstack(ycovid2)
    y_test2 = np.expand_dims(y_test2, axis=3)
    y_test2[y_test2 > 0] = 1
    
    scaler         = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test2_scaled = scaler.transform(X_test2.reshape(-1, X_test2.shape[-1])).reshape(X_test2.shape)
    
    test2_num = np.random.rand(X_test2_scaled.shape[0]).argsort()
    X_test2 = np.take(X_test2_scaled, test2_num , axis=0, out=X_test2_scaled)
    y_test2 = np.take(y_test2, test2_num, axis=0, out=y_test2)
    
    return X_test2, y_test2

#%%

def historyPlot(results, checkpoint_savepath, i, doSave):

    plt.figure(figsize=(8, 8))
    plt.title("Learning curve")
    plt.plot(results.history["loss"], label="loss")
    plt.plot(results.history["val_loss"], label="val_loss")
    plt.plot(np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("log_loss")
    plt.legend()
    if doSave:
        plt.savefig(checkpoint_savepath+'/fold_%d.jpg' %(i+1), quality=90)
    plt.show()