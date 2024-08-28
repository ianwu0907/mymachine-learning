# Bearing_Diagnostics
This project will keep updating.
- This is a project to diagnose bearing faults using machine learning techniques. The demo files uses ***Case Western Reserve University Bearing Data Set***. The full dataset can be downloaded from the ***data.zip*** file in the repository.
- Before running the model_train.ipynb file, please make sure that you have ran the **data_create.ipynb** file to create the train_data and label datasets.
- Then run the **model_train.ipynb** file to train the models.
- On the first upload of this repository, the **model_train.ipynb** file contains RandomForest and SVM models to diagnose bearing faults. I will update the file with more traditional machine-learning models in the future.
- It is also in plan to implement deep learning models(CNN, LSTM, Transformer...) to diagnose bearing faults in the future.
## Before running the model_train.ipynb file
- Please make sure you have installed the following libraries:
```
numpy,tensorflow,sckit-learn,pandas,matplotlib,pytorch,scipy,collections
```
- If you have not installed the libraries, you can install them using the following command:
```
pip install numpy tensorflow sckit-learn pandas matplotlib pytorch scipy collections
```
---
# Updates
- 2024/7/31 added machine-learning models to classify 10 types of faults with CWRU dataset. ***model_train_10classes.ipynb*** and ***data_create_10classes.py*** are added to the repository.
- 2024/8/1  added data pre-processing and feature extraction file ***data_create_CWRU_FFT.ipynb*** to the repository which performs fast-fourier transformation and then Envelope Spectrum Analysis on the CWRU dataset, extracting 3 feature frequency data.
- 2024/8/1  added file ***model_train_FFT.ipynb*** to the repository which trains the machine-learning models on the feature frequency data extracted from the CWRU dataset. 
- 2024/8/5  updated ***model_train.ipynb***, added CNN and CNN+ResNet models to diagnose bearing faults. With excellent performance.
- 2024/8/28 updated ***datacreate_IMS.ipynb***,  the data pre-processing  for the IMS dataset.
- 2024/8/28 updated ***IMS_2D_4class.ipynb***, the data created is firstly transformed into 2D images, then models are trained on the 2D images. !!!***CWRU Dataset*** can be also applied to this notebook, using the ***data_create.ipynb*** to create the data and a little amendment to the "data_load" function in ***IMS_2D_4class.ipynb***.
---
# Author's Note
- 2024/7/31 The 10 class classification with traditional machine learning technique SVM has a better overall performance.
- 2024/8/28 It is tested feasible, that the models will produce a better performance with 2d transformation method 