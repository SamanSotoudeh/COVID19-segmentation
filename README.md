# COVID-19 Segmentation

### This repo contains the implelmentation of the following paper available using this link: https://brieflands.com/articles/iranjradiol-117992.html
Sotoudeh-Paima, S., Hasanzadeh, N., Bashirgonbadi, A., Aref, A., Naghibi, M., Zoorpaikar, M., ... & Soltanian-Zadeh, H. (2022). A Multi-centric Evaluation of Deep Learning Models for Segmentation of COVID-19 Lung Lesions on Chest CT Scans. Iranian Journal of Radiology, 19(4).

__Objectives:__ This study aimed to evaluate the performance and generalizability of deep learning-based models for the automated segmentation of COVID-19 lung lesions.

### This repo was used for analysis and has been shared as part of the following paper available using this link: <link-will-be-provided-here>
"citation-will-be-provided-here"

__Objectives:__ This study investigates the performance of an AI-aided quantification model in predicting the clinical outcomes of hospitalized subjects with COVID-19 and compares it with radiologists’ performance.

### The dataset used in this paper is publicly available using this link: "path-will-be-provided-here"

#### Please cite both papers in case you have used our work in your research/project.

## Code Explanation
The code package consists of four separate .py files:
- dataloader.py: loads the data in .nii format
- helpers.py: functions that are used in the main code
- params.py: all parameters to run the AI model
- train.py: main code

## Steps to Run the Code
- download the dataset
- set the main path in 'params.py' (line 9) and relative path of datasets in 'train.py' (lines 73 to 75).
- change the hyperparameters or model structure/backbone as needed.
- fit the model by running the 'train.py' code and monitor the training/validation accuracy
- evaluation and testing using the saved model

## Libraries with version number
numpy 1.22.4  
segmentation-models 1.0.1   
matplotlib 3.5.2  
scikit-learn 1.1.1  
scikit-image 0.19.3  
SimpleITK 2.1.1.2  
opencv-python 4.5.5.62  
keras 2.9.0  
tensorflow 2.9.1  
