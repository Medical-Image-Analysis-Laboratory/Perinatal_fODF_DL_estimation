# Fiber ODF estimation with a deep neural network

Code for paper: Robust Estimation of the Microstructure of the Early Developing Brain Using Deep Learning By: Hamza Kebiri, Ali Gholipour, Rizhong Lin, Lana Vasung and Davood Karimi*, Meritxell Bach Cuadra* (*: Equal contribution)

Accepted at Medical Image Computing and Computer Assisted Intervention – MICCAI 2023.

If you have any questions about the code, please email hamza.kebiri@unil.ch

Files included:

fod_cnn_genData.py to generate the input data, ground truth data and CSD baseline from dHCP data
fod_cnn_train.py to train the CNN to learn FOD predictions from input data
fod_cnn_test.py to test the CNN 
fod_cnn_genDataGS.py to generate the two gold standards (GS1 and GS2)

The MLP folders contains TrainMLP.py to train the MLP model of Karimi et al., Neuroimage, 2021

dk_aux.py, dk_model.py, crl_aux and dk_seg.py include functions (developped by davood.karimi@childrens.harvard.edu) that are used by the files above.
