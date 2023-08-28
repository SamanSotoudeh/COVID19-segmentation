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
    - imageSize
    size of the output image
    - closing
    whether to apply morphological closing on the label mask
    
Return:
    - X
    3D data array of CT volume
    - y
    3D data array of label volume
'''

def covidLoader(dataPath, imageSize, closing):
    
    imageSize = 256 
    X         = []
    y         = []
    count     = 0
    
    # morphological operation (closing)
    closing   = 1
    kernel    = np.ones((5,5),np.uint8)

    for cases in os.listdir(dataPath):
        for files in os.listdir(dataPath+cases):
            if (files.endswith('.nii.gz')) and ('ct' in files):
                img_file = sitk.ReadImage(dataPath + cases + '/' + files)
                img_file = sitk.GetArrayFromImage(img_file)
                img_file = np.int16(img_file)
                img_file = resize(img_file, (imageSize, imageSize, img_file.shape[2]), mode = 'constant', preserve_range = True)
                img_file = np.rot90(img_file, 1)
                img_file = np.flip(img_file, 1)
    
                if img_file is not None:
                    img_arr = np.asarray(img_file)
                    X.append(img_arr)
                
            elif (files.endswith('.nii.gz')) and ('label' in files):
                img_file = sitk.ReadImage(dataPath + cases + '/' + files)
                img_file = sitk.GetArrayFromImage(img_file)
                img_file = resize(img_file, (imageSize, imageSize, img_file.shape[2]), mode = 'constant', preserve_range = True)
                img_file = np.rot90(img_file, 1)
                img_file = np.flip(img_file, 1)
                if closing:
                    img_file = cv2.morphologyEx(img_file, cv2.MORPH_CLOSE, kernel)
    
                if img_file is not None:
                    img_arr = np.asarray(img_file)
                    y.append(img_arr)  
            
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
    
    for files in os.listdir(dataPath):
        if (files.endswith('.nii.gz')) and ('ct' in files):
            img_file = sitk.ReadImage(dataPath + '/' + files)
            img_file = sitk.GetArrayFromImage(img_file)
            img_file = np.int16(img_file)
            img_file = resize(img_file, (imageSize, imageSize, img_file.shape[2]), mode = 'constant', preserve_range = True)
            img_file = np.rot90(img_file, 1)
            img_file = np.flip(img_file, 1)
    
            if img_file is not None:
                X.append(img_file)
                y.append(np.zeros(np.shape(img_file)))
            
            print(count)
            count+= 1
            
    return X, y