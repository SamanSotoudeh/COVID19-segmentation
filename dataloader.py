'''
Dataloader functions
'''

#%%

import os     
import cv2
import numpy as np
import SimpleITK as sitk
from skimage.transform import resize

#%%

'''
Loads a COVID-19 dataset.
Inputs:
    - dataPath
    path to the CT folder
    - labelPath
    path to the labels folder
    
Return:
    - X
    3D data array of CT volume
    - y
    3D data array of label volume
'''

def covidLoader(dataPath, labelPath):
    
    imageSize = 256 
    X         = []
    y         = []
    count     = 0
    kernel    = np.ones((5,5),np.uint8)

    for file in os.listdir(labelPath):
        if file.endswith('.img'):
            img_file = sitk.ReadImage(labelPath + file)
            img_file = sitk.GetArrayFromImage(img_file)
            img_file = resize(img_file, (len(img_file), imageSize, imageSize), mode = 'constant', preserve_range = True)
            img_file = 1.0 * (img_file >= 100)
            img_file = cv2.morphologyEx(img_file, cv2.MORPH_CLOSE, kernel)
    
            if img_file is not None:
                img_arr = np.asarray(img_file)
                y.append(img_arr)
            
    for file in os.listdir(dataPath):
        if file.endswith('.img'):
            img_file = sitk.ReadImage(dataPath + file)
            img_file = sitk.GetArrayFromImage(img_file)
            img_file = resize(img_file, (len(img_file), imageSize, imageSize, 1), mode = 'constant', preserve_range = True)
                
            if img_file is not None:
                img_arr = np.asarray(img_file)
                X.append(img_arr)  
            
        print(count)
        count+= 1

    return X, y

#%%

'''
Loads a normal dataset.
Inputs:
    - dataPath
    path to the CT folder
    * no labelPath because label is a 3D array of all zeros.
    
Return:
    - X
    3D data array of CT volume
    - y
    3D data array of label volume
'''

def normalLoader(dataPath):

    imageSize = 256

    X = []
    y = []
    count = 0    
    
    for file in os.listdir(dataPath):
        
        if file.endswith('.img'):
            img_file = sitk.ReadImage(dataPath + file)
            img_file = sitk.GetArrayFromImage(img_file)
            img_file = resize(img_file, (len(img_file), imageSize, imageSize, 1), mode = 'constant', preserve_range = True)
    
            if img_file is not None:
                img_arr = np.asarray(img_file)
                X.append(img_arr)
                y.append(np.zeros(np.shape(img_arr)[0:-1]))
            
            print(count)
            count+= 1
            
    return X, y