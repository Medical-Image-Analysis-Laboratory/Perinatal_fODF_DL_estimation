#/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ch235786
"""

import os
import numpy as np
import nibabel as nib
from dipy.core.sphere import Sphere
from dipy.reconst.shm import sf_to_sh
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import dk_model

###############################################################################

base_dir = '/home/ch051094/Desktop/DataDWI2FOD/'
images_dir = base_dir + 'images_test/'

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

##############################################################################

N_grad = 6 # because of SH2, otherwise: 15 for SH4
n_sig = N_grad
n_tar = 45 # SH8 of FOD

#Network specifications
LX = LY = LZ = 16  # 16 #48
n_feat_0 = 36  # 12
depth = 3
ks_0 = 3
X = tf.placeholder("float32", [None, LX, LY, LZ, n_sig])
Y = tf.placeholder("float32", [None, LX, LY, LZ, n_tar])
learning_rate = tf.placeholder("float")
p_keep_conv = tf.placeholder("float")
Y_pred, _ = dk_model.davood_net(X, ks_0, depth, n_feat_0, n_sig, n_tar, p_keep_conv, bias_init=0.001)

gpu_ind = 0
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ind)
saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
keep_test = 1.0


invBvecs = False
mapInputToSH = True
sh_order = 2
subjects_15 = ['CC00071XX06', 'CC00119XX12', 'CC00122XX07', 'CC00202XX04', 'CC00207XX09', 'CC00407BN11', 'CC00465XX12',
               'CC00466BN13', 'CC00467XX14', 'CC00476XX15', 'CC00478XX17', 'CC00484XX15', 'CC00492BN15', 'CC00500XX05',
               'CC00512XX09']

#Load checkpoint and run network on test subjects subjects_15

ckpt_path = '/home/ch051094/Desktop/DataDWI2FOD_shard/DL_90sub_EqualSampling9-TestSet15-MergedTrueReal/train_06/thumbs/model_076_1221.ckpt'
saver.restore(sess, ckpt_path)

for test_subject in subjects_15:
    test_dir = images_dir + test_subject + '/'
    if os.path.exists(test_dir + 'd_mri_subset_'+str(6)+'.nii.gz'):
        dmri_vol= nib.load( test_dir + 'd_mri_subset_'+str(6)+'.nii.gz' )
        affine= dmri_vol.affine
        dmri_vol= dmri_vol.get_data()
        dmri_vol = np.nan_to_num(dmri_vol, posinf=1, neginf=0)
        dmri_vol = dmri_vol*10
        if mapInputToSH:
            b_vecs_selected_curr = np.loadtxt(test_dir +'b_vecs_selected.txt')
            b_vecs_selected_curr = np.asarray(b_vecs_selected_curr).T
            if invBvecs:
                print("bvecs shape", b_vecs_selected_curr.shape)
                for i in range(b_vecs_selected_curr.shape[0]):
                    b_vecs_selected_curr[i,0] = -b_vecs_selected_curr[i,0]
                    b_vecs_selected_curr[i, 1] = -b_vecs_selected_curr[i, 1]
                    b_vecs_selected_curr[i, 2] = b_vecs_selected_curr[i, 2]
            sphere_bvecs = Sphere(xyz=b_vecs_selected_curr)
            dmri_vol = sf_to_sh(dmri_vol, sphere=sphere_bvecs, sh_order=sh_order, basis_type='tournier07')

        SX, SY, SZ= dmri_vol.shape[:3]
        LX= LY= LZ = 16
        test_shift= LX//3
        lx_list= np.squeeze( np.concatenate(  (np.arange(0, SX-LX, test_shift)[:,np.newaxis] , np.array([SX-LX])[:,np.newaxis] )  ) .astype(np.int) )
        ly_list= np.squeeze( np.concatenate(  (np.arange(0, SY-LY, test_shift)[:,np.newaxis] , np.array([SY-LY])[:,np.newaxis] )  ) .astype(np.int) )
        lz_list= np.squeeze( np.concatenate(  (np.arange(0, SZ-LZ, test_shift)[:,np.newaxis] , np.array([SZ-LZ])[:,np.newaxis] )  ) .astype(np.int) )
        LXc= LYc= LZc= 6
        y_s = np.zeros((SX, SY, SZ, n_tar))
        y_c = np.zeros((SX, SY, SZ))
        for lx in lx_list:
            for ly in ly_list:
                for lz in lz_list:
                    if np.max(dmri_vol[ lx+LXc:lx+LX-LXc, ly+LYc:ly+LY-LYc, lz+LZc:lz+LZ-LZc,0]) > 0:
                        batch_x = dmri_vol[ lx:lx + LX, ly:ly + LY, lz:lz + LZ, :].copy()
                        batch_x= batch_x[np.newaxis,:]
                        pred_temp = sess.run(Y_pred,  feed_dict={X: batch_x, p_keep_conv: keep_test})
                        y_s[lx:lx + LX, ly:ly + LY, lz:lz + LZ, :] += pred_temp[0, :, :, :, :]
                        y_c[lx:lx + LX, ly:ly + LY, lz:lz + LZ] += 1
        y_s= y_s/ np.tile( y_c[:,:,:,np.newaxis]+0.00001, [1, 1, 1, n_tar])
        y_s/= 10
        #for i in range(0, y_s.shape[-1]):
        #    y_s[...,i] /= abs_vals[i]
        y_s = nib.Nifti1Image(y_s, affine)
        nib.save(y_s, test_dir + 'DL_90sub_EqualSampling9-TestSet15.nii.gz' )
