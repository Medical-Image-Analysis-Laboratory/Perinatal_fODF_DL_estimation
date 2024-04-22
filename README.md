# Fiber ODF estimation with a deep neural network

Code for paper: Robust Estimation of the Microstructure of the Early Developing Brain Using Deep Learning, by: Hamza Kebiri, Ali Gholipour, Rizhong Lin, Lana Vasung and Davood Karimi*, Meritxell Bach Cuadra* (*: Equal contribution) and Deep learning microstructure estimation of developing brains from diffusion MRI: a newborn and fetal study, by Hamza Kebiri, Ali Gholipour, Rizhong Lin, Lana Vasung, Camilo Calixto, Željka Krsnik and Davood Karimi*, Meritxell Bach Cuadra* (*: Equal contribution). 

Accepted at Medical Image Computing and Computer Assisted Intervention – MICCAI 2023 (conference) and Medical Image Analysis (journal).

Links: https://link.springer.com/chapter/10.1007/978-3-031-43990-2_28 (MICCAI) and TBD (MedIA)

If you have any questions about the code, please email hamza.kebiri@unil.ch

Files included:

fod_cnn_genData.py to generate the input data, ground truth data and CSD baseline from dHCP data

fod_cnn_train.py to train the CNN to learn FOD predictions from input data

fod_cnn_test.py to test the CNN

fod_cnn_genDataGS.py to generate the two gold standards (GS1 and GS2)

The MLP folder contains TrainMLP.py to train the MLP model (by rizhong.lin@epfl.ch) of Karimi et al., Neuroimage, 2021

dk_aux.py, dk_model.py, crl_aux and dk_seg.py include auxiliary functions (developped by davood.karimi@childrens.harvard.edu) that are used by the scripts above.

Please note that the network code has been reimplemented by Rizhong Lin (rizhong.lin@epfl.ch) with a newer version of TensorFlow at https://github.com/Medical-Image-Analysis-Laboratory/dl_fiber_domain_shift/tree/main/DeepLearning/kebiri_robust_2023
