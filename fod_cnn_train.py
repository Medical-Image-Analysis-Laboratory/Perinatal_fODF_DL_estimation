#/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 08:25:14 2022

@authors: ch209389, ch235786
"""

import os
from os import listdir
from os.path import isdir, join
import numpy as np
import nibabel as nib
import pandas as pd
import time
import h5py
import dipy.core.sphere as dipysphere
from dipy.core.sphere import Sphere
from dipy.reconst.shm import sf_to_sh

import SimpleITK as sitk
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


import dk_model
import crl_aux
import dk_seg

###############################################################################


# Training data

dhcp_dir = '/home/ch051094/Desktop/dhcp_4_hamza/'
subj_info = pd.read_csv(dhcp_dir + 'participants.tsv', delimiter='\t')

# to save the images that will be used for model training
base_dir = '/home/ch051094/Desktop/DataDWI2FOD/'
images_dir = base_dir + 'images/'
training_dir = base_dir + 'DL_90sub_EqualSampling9-TestSet15/'
mapInputToSH = True

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   #TO REMOVE FOR TRAINING !!!!!!!!!!!!!!!!!!!!!!!!!!!!

# os.makedirs(images_dir, exist_ok=True)
os.makedirs(training_dir, exist_ok=True)

n_max_subjects = 100  # 73

desired_spacing = (1.0, 1.0, 1.0)

# number of measurement subsets

# N_grad_1 = 32 #b400
# Xp, Yp, Zp= crl_aux.distribute_on_hemisphere_spiral(N_grad_1)
# sphere_grad = dipysphere.Sphere(Xp, Yp, Zp)
# v_grad_1, _ = sphere_grad.vertices, sphere_grad.faces
#
# N_grad_2 = 44 #b1000
# Xp, Yp, Zp= crl_aux.distribute_on_hemisphere_spiral(N_grad_2)
# sphere_grad = dipysphere.Sphere(Xp, Yp, Zp)
# v_grad_2, _ = sphere_grad.vertices, sphere_grad.faces
#
# N_grad_3 = 64 #b2600
# Xp, Yp, Zp= crl_aux.distribute_on_hemisphere_spiral(N_grad_3)
# sphere_grad = dipysphere.Sphere(Xp, Yp, Zp)
# v_grad_3, _ = sphere_grad.vertices, sphere_grad.faces
#
# N_grad_list= [ N_grad_1, N_grad_2, N_grad_3 ]
# V_grad_list= [ v_grad_1, v_grad_2, v_grad_3 ]

N_grad_1 = 6  # 12 #b400
Xp, Yp, Zp = crl_aux.optimized_six()
sphere_grad = dipysphere.Sphere(Xp, Yp, Zp)
v_grad_1, _ = sphere_grad.vertices, sphere_grad.faces

N_grad_list = [N_grad_1]
V_grad_list = [v_grad_1]

temp = []
start_time = time.time()

##############################################################################
# Create data tensors for machine learning   1
##############################################################################
subjects_15 = ['CC00071XX06', 'CC00119XX12', 'CC00122XX07', 'CC00202XX04', 'CC00207XX09', 'CC00407BN11', 'CC00465XX12',
               'CC00466BN13', 'CC00467XX14', 'CC00476XX15', 'CC00478XX17', 'CC00484XX15', 'CC00492BN15', 'CC00500XX05',
               'CC00512XX09']
#subjects_15 = subjects_15[-5:]  # subjects_15[:10] last 5 have these indices [67,68,69,70,72]

subject_results_dirs = [os.path.join(images_dir, d) for d in listdir(images_dir) if isdir(join(images_dir, d))]

subject_results_dirs.sort()

N_grad = 6
nInstances = 5  # put to 1 if no need for condition number selection

n_sig = N_grad  # 15 #because of SH4, otherwise: N_grad
n_tar = 45  # SH6
n_data = n_max_subjects * nInstances
sx, sy, sz = 150, 150, 96

X_data = np.zeros((n_data, sx, sy, sz, n_sig), np.float16)
Y_gold = np.zeros((n_data, sx, sy, sz, n_tar), np.float16)
# Y_base = np.zeros((n_data, sx, sy, sz, n_tar), np.float32)
XY_mask = np.zeros((n_data, sx, sy, sz), np.int16)

i_data = -1

allVolumes = True
if not os.path.exists(base_dir + 'data_' + str(N_grad) + '_notTestSetDifferentBvecsCondNumber2.h5'):
    print("Preparing h5 dataset")
    for subject_results_dir in subject_results_dirs[:n_max_subjects]:
        for instance in range(nInstances):
            if subject_results_dir.split('/')[-2] not in subjects_15:
                if not allVolumes:
                    temp = subject_results_dir + 'd_mri_subset_' + str(N_grad) + '.nii.gz'
                    temp = nib.load(temp)
                    x_data = temp.get_data()
                elif allVolumes and instance == 0:
                    # read the denoised volume
                    dmri_volume = sitk.ReadImage(subject_results_dir + 'dwi_debiased.nii.gz')
                    dmri_volume = sitk.GetArrayFromImage(dmri_volume)
                    dmri_volume = np.transpose(dmri_volume, [3, 2, 1, 0])

                    # interpolate the data

                    brain_mask_img = sitk.ReadImage(subject_results_dir + 'mask_orig.nii.gz')
                    original_spacing = brain_mask_img.GetSpacing()
                    original_size = brain_mask_img.GetSize()

                    n_dwi = dmri_volume.shape[-1]
                    SX, SY, SZ = [int(original_size[i] / desired_spacing[i] * original_spacing[i]) for i in range(3)]

                    d_img0 = dmri_volume.copy()
                    dmri_volume = np.zeros((SX, SY, SZ, n_dwi), np.float32)
                    for i in range(n_dwi):
                        temp = d_img0[:, :, :, i].copy()
                        temp = np.transpose(temp, [2, 1, 0])
                        temp = sitk.GetImageFromArray(temp)
                        temp.SetDirection(brain_mask_img.GetDirection())
                        temp.SetOrigin(brain_mask_img.GetOrigin())
                        temp.SetSpacing(brain_mask_img.GetSpacing())
                        temp = dk_seg.resample3d(temp, original_spacing, desired_spacing, sitk.sitkBSpline)
                        temp = sitk.GetArrayFromImage(temp)
                        temp = np.transpose(temp, [2, 1, 0])
                        dmri_volume[:, :, :, i] = temp
                    del d_img0

                if mapInputToSH:
                    b_vecs = np.loadtxt(
                        subject_results_dir + 'bvecs.bvec')  # change this to load random bvecs! and loop until cond number good
                    b_vals = np.loadtxt(subject_results_dir + 'bvals.bval')
                    nbvecs = 6
                    cond_num = 2
                    bval = 1000
                    b_vecs_selected_curr, idx_bvecs = crl_aux.randomSelectBvecs(b_vecs, nbvecs, b_vals, bval, cond_num)
                    # b_vecs_selected_curr = np.loadtxt(subject_results_dir + 'b_vecs_selected.bvec') # change this to load random bvecs! and loop until cond number good
                    dmri_volume[:, :, :, b_vals == bval]
                    x_data = dmri_volume[:, :, :, idx_bvecs]
                    sphere_bvecs = Sphere(xyz=np.asarray(b_vecs_selected_curr).T)
                    x_data = sf_to_sh(x_data, sphere=sphere_bvecs, sh_order=2, basis_type='tournier07')

                # temp= subject_results_dir + 'csd_fodf_mrtrix_1000_' +  str(N_grad) + '.nii.gz'
                # temp= nib.load( temp )
                # y_base= temp.get_data()

                # temp= subject_results_dir.replace('_withBvecs_SH', '') + 'wm.nii.gz' #hack to take WM from other dir
                temp = subject_results_dir + 'wm.nii.gz'
                temp = nib.load(temp)
                y_gold = temp.get_data()

                temp = subject_results_dir + 'npeaks_HD_WM.nii.gz'
                temp = nib.load(temp)
                xy_mask = temp.get_data()

                files_ok = True

                # except:

                # print('Data not processed for subject: ', subject_results_dir)

                if files_ok:
                    i_data += 1
                    X_data[i_data, :, :, :, :] = x_data.copy()
                    Y_gold[i_data, :, :, :, :] = y_gold.copy()
                    # Y_base[i_data,:,:,:,:]= y_base.copy()
                    XY_mask[i_data, :, :, :] = xy_mask.copy()

    X_data = X_data[:i_data + 1, :, :, :, :]
    Y_gold = Y_gold[:i_data + 1, :, :, :, :]
    # Y_base= Y_base[:i_data+1,:,:,:,:]
    XY_mask = XY_mask[:i_data + 1, :, :, :]

    assert (np.sum(np.isnan(X_data)) == 0)
    assert (np.sum(np.isnan(Y_gold)) == 0)
    # assert( np.sum(np.isnan(Y_base)) == 0 )
    assert (np.sum(np.isnan(XY_mask)) == 0)

    mk_ext = np.where(np.sum(X_data != 0, axis=0) > 0)
    border = 5  # 5
    x_beg, x_end = max(mk_ext[0].min() - border, 0), min(mk_ext[0].max() + border, sx)
    y_beg, y_end = max(mk_ext[1].min() - border, 0), min(mk_ext[1].max() + border, sy)
    z_beg, z_end = max(mk_ext[2].min() - border, 0), min(mk_ext[2].max() + border, sz)

    X_data = X_data[:, x_beg:x_end, y_beg:y_end, z_beg:z_end, :]
    Y_gold = Y_gold[:, x_beg:x_end, y_beg:y_end, z_beg:z_end, :]
    # Y_base = Y_base[:, x_beg:x_end, y_beg:y_end, z_beg:z_end,:]
    XY_mask = XY_mask[:, x_beg:x_end, y_beg:y_end, z_beg:z_end]

    h5f = h5py.File(base_dir + 'data_' + str(N_grad) + '_notTestSetDifferentBvecsCondNumber2.h5', 'w')
    h5f['X_data'] = X_data
    h5f['Y_gold'] = Y_gold
    # h5f['Y_base']= Y_base
    h5f['XY_mask'] = XY_mask

    h5f.close()

    del X_data, Y_gold, XY_mask  # , Y_base

h5f = h5py.File(base_dir + 'data_' + str(N_grad) + '_notTestSet.h5', 'r')
# h5f = h5py.File( base_dir + 'data_' + str(N_grad) +'_notTestSet.h5' ,'r')

n_all = h5f['X_data'].shape[0]
np.random.seed(0)
p = np.arange(n_all)

n_all = 85 * nInstances
p2 = np.concatenate([p[0:67 * nInstances], p[73 * nInstances:]])
p = p2

np.random.shuffle(p)
p_test = p[:int(round(0.25 * n_all))]
p_train = p[int(round(0.25 * n_all)):]
p_test.sort()
p_train.sort()
X_test = h5f['X_data'][p_test, :, :, :, :]
Y_test = h5f['Y_gold'][p_test, :, :, :, :]
X_train = h5f['X_data'][p_train, :, :, :, :]
Y_train = h5f['Y_gold'][p_train, :, :, :, :]

# mask_train = h5f['XY_mask'][p_train,:,:,:]
# mask_test = h5f['XY_mask'][p_test,:,:,:]
# mask_train = (mask_train/85).astype(int) #85 comes from 3 peaks to uint
# mask_test = (mask_test/85).astype(int)

h5f2 = h5py.File(base_dir + 'data_' + str(N_grad) + '_notTestSet_withNpeaksMasks.h5', 'r')
mask_train = h5f2['XY_mask'][p_train, :, :, :]
mask_test = h5f2['XY_mask'][p_test, :, :, :]

# To remove if not using n Instances
mask_both = h5f2['XY_mask'][:, :, :, :]
mask_both_nInstances = np.empty([n_all, X_train.shape[0], X_train.shape[1], X_train.shape[2]])

for i in range(n_all):
    for j in range(nInstances):
        mask_both_nInstances[j + i * nInstances, :, :, :] = mask_both[i, :, :, :]

mask_train = mask_both_nInstances[p_train, :, :, :]
mask_test = mask_both_nInstances[p_test, :, :, :]

# To remove

mask_train = (mask_train / 85).astype(int)  # 85 comes from 3 peaks to uint
mask_test = (mask_test / 85).astype(int)

h5f.close()
h5f2.close()

###########################################################################################
# gt,aff = load_nifti('/home/ch051094/Desktop/DataDWI2FOD/images/CC00071XX06/wm.nii.gz')
#
# abs_vals = []
# first_coeff = gt[..., 0]
# abs_vals.append(np.sum(abs(first_coeff)))
# for i in range(1, gt.shape[-1]):
#     curr_coeff = gt[..., i]
#     abs_vals.append(10*(abs_vals[0]/np.sum(abs(curr_coeff))))
#     print(abs_vals[i])
#     Y_train[...,i] *= abs_vals[i]  # 10
#     Y_test[...,i] *= abs_vals[i] # 10
# abs_vals[0] = 10
# Y_train[..., 0] *= abs_vals[0]  # 10
# Y_test[..., 0] *= abs_vals[0]  # 10

Y_train *= 10
Y_test *= 10

n_train, SX, SY, SZ, n_sig = X_train.shape

M_train = X_train[:, :, :, :, 0] != 0
M_test = X_test[:, :, :, :, 0] != 0

gpu_ind = 0
L_Rate = 5.0e-5  # 1.0e-4 5.0e-6 #5.0e-5 #

train_dir = training_dir + 'train_06/'
model_dir = train_dir + 'model/'
thumbs_dir = train_dir + 'thumbs/'
os.makedirs(model_dir)
os.makedirs(thumbs_dir)

LX = LY = LZ = 16  #8 #32 #48
test_shift = LX // 3
lx_list = np.squeeze(
    np.concatenate((np.arange(0, SX - LX, test_shift)[:, np.newaxis], np.array([SX - LX])[:, np.newaxis])).astype(
        np.int))
ly_list = np.squeeze(
    np.concatenate((np.arange(0, SY - LY, test_shift)[:, np.newaxis], np.array([SY - LY])[:, np.newaxis])).astype(
        np.int))
lz_list = np.squeeze(
    np.concatenate((np.arange(0, SZ - LZ, test_shift)[:, np.newaxis], np.array([SZ - LZ])[:, np.newaxis])).astype(
        np.int))
LXc = LYc = LZc = 6  # 6 #18

prop_nonzero_patch = 0.33
balance_factor = 1 // prop_nonzero_patch

n_feat_0 = 36  # 12
depth = 3  # set to 3 if decrease LX
ks_0 = 3

X = tf.placeholder("float32", [None, LX, LY, LZ, n_sig])
Y = tf.placeholder("float32", [None, LX, LY, LZ, n_tar])
learning_rate = tf.placeholder("float")
p_keep_conv = tf.placeholder("float")

Y_pred, _ = dk_model.davood_net(X, ks_0, depth, n_feat_0, n_sig, n_tar, p_keep_conv, bias_init=0.001)

'''
#Loss function design
test_dir = '/home/ch051094/Desktop/DataDWI2FOD/images/CC00120XX05/' #random subject taken because ratios almost same for all
dmri_coeffs = nib.load(test_dir + 'wm.nii.gz').get_data()

abs_vals = []
first_coeff = dmri_coeffs[..., 0]
abs_vals.append(np.sum(abs(first_coeff)))
weights = np.zeros((n_max_subjects,LX, LY, LZ, n_tar))
weights[..., 0] = 1
# for i in range(1, n_tar):
#     curr_coeff = dmri_coeffs[..., i]
#     abs_vals.append(abs_vals[0]/np.sum(abs(curr_coeff)))
#     print(abs_vals[i])
#     weights[..., i] = abs_vals[i]

for i in range(1, n_tar):
    if i < 6:
        k = 2
    elif i <15:
        k = 4
    elif i <28:
        k = 8
    else:
        k = 20
    weights[..., i] = k
    print(k)

print("Weight shape", weights.shape)
abs_vals[0] = 1
weights = weights.astype(np.float32)
weights = tf.constant(weights,dtype='float32')
#tf.compat.v1.assign(a,weights)
#tf.cast(a,tf.float32)

#
cost= tf.reduce_mean( tf.pow( (Y_pred- Y)*weights, 2 ) )
#cost = tf.reduce_mean( tf.abs( (Y_pred- Y)*weights)  )

'''
# cost= tf.reduce_mean( tf.abs( Y_pred- Y ) )
cost = tf.reduce_mean(tf.pow(Y_pred - Y, 2))
#


optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ind)
saver = tf.train.Saver(max_to_keep=50)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

i_global = 0
i_gross = 0
best_test = np.inf

i_eval = -1

batch_size = 27  # 3 #1
n_epochs = 10000 * 10

test_interval = 10000 // 2

keep_train = 0.9
keep_test = 1.0

additional_save_ind = [1, 10, 300, 500]

epochs_w_no_progress = 0

for epoch_i in range(n_epochs):

    # for i in range(n_train // batch_size):
    for subj_i in range(n_train):

        # Balance sampling of single v.s. multiple peaks PER BATCH
        batch_x = np.empty([batch_size, LX, LY, LZ, X_train.shape[4]])
        batch_y = np.empty([batch_size, LX, LY, LZ, Y_train.shape[4]])

        balance_train = 0
        for k in range(batch_size):

            if not balance_train % balance_factor == 0:
                while True:
                    x_i = np.random.randint(SX - LX - 1)
                    y_i = np.random.randint(SY - LY - 1)
                    z_i = np.random.randint(SZ - LZ - 1)

                    batch_m = M_train[subj_i:subj_i+1, x_i:x_i + LX, y_i:y_i + LY, z_i:z_i + LZ].copy()
                    conditionMin = np.min(batch_m[:, LXc:LX - LXc, LYc:LY - LYc, LZc:LZ - LZc]) > 0

                    condition1 = mask_train[subj_i, x_i + LX // 2, y_i + LY // 2, z_i + LZ // 2] == 1

                    if condition1 and conditionMin:
                        batch_x[k, ...] = X_train[subj_i, x_i:x_i + LX, y_i:y_i + LY, z_i:z_i + LZ, :].copy()
                        batch_y[k, ...] = Y_train[subj_i, x_i:x_i + LX, y_i:y_i + LY, z_i:z_i + LZ, :].copy()

                        break

            else:
                while True:

                    x_i = np.random.randint(SX - LX - 1)
                    y_i = np.random.randint(SY - LY - 1)
                    z_i = np.random.randint(SZ - LZ - 1)

                    batch_m = M_train[subj_i:subj_i+1, x_i:x_i + LX, y_i:y_i + LY, z_i:z_i + LZ].copy()
                    conditionMin = np.min(batch_m[:, LXc:LX - LXc, LYc:LY - LYc, LZc:LZ - LZc]) > 0

                    condition1 = mask_train[subj_i, x_i + LX // 2, y_i + LY // 2, z_i + LZ // 2] == 2

                    if condition1 and conditionMin:
                        batch_x[k, ...] = X_train[subj_i, x_i:x_i + LX, y_i:y_i + LY, z_i:z_i + LZ, :].copy()
                        batch_y[k, ...] = Y_train[subj_i, x_i:x_i + LX, y_i:y_i + LY, z_i:z_i + LZ, :].copy()

                        break
            balance_train += 1

            if subj_i == n_train - 1:
                batch_x = batch_x[:k + 1, ...]
                batch_y = batch_y[:k + 1, ...]
                break


        # try:
        if True:  #
            sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y, learning_rate: L_Rate, p_keep_conv: keep_train})
            i_global += 1

        i_gross += 1
        # except ValueError:
        #   print('Subject '+str(i)+' not processed')
        #      pass
        # test

        

    if i_gross % test_interval == 0 or i_gross in additional_save_ind:

        i_eval += 1

        print('\n' * 2)
        print(epoch_i, i, i_global, i_gross)
        # print(Cost.mean(), Cost.std())

        strd = 5
        loss_c = np.zeros((X_train.shape[0] // strd + 1, 2))
        # generate a random starting point between 0 and strd
        start_point = np.random.randint(strd)

        for i_c in range(start_point, X_train.shape[0], strd):

            y_s = np.zeros((SX, SY, SZ, n_tar))
            y_c = np.zeros((SX, SY, SZ))

            for lx in lx_list:
                for ly in ly_list:
                    for lz in lz_list:

                        if np.min(M_train[i_c:i_c + 1, lx + LXc:lx + LX - LXc, ly + LYc:ly + LY - LYc,
                                  lz + LZc:lz + LZ - LZc]) > 0:
                            batch_x = X_train[i_c:i_c + 1, lx:lx + LX, ly:ly + LY, lz:lz + LZ, :].copy()

                            pred_temp = sess.run(Y_pred, feed_dict={X: batch_x, p_keep_conv: keep_test})

                            y_s[lx:lx + LX, ly:ly + LY, lz:lz + LZ, :] += pred_temp[0, :, :, :, :]
                            y_c[lx:lx + LX, ly:ly + LY, lz:lz + LZ] += 1

            y_s = y_s / np.tile(y_c[:, :, :, np.newaxis] + 0.00001, [1, 1, 1, n_tar])

            # batch_x = X_train[i_c, :, :, :, :].copy()
            batch_y = Y_train[i_c, :, :, :, :].copy()
            # batch_y_n = Y_base_train[i_c, :, :, :, :].copy()
            batch_M = M_train[i_c, :, :, :].copy()

            temp = np.abs(y_s - batch_y)[batch_M == 1]
            loss_c[i_c // strd, 0] = np.mean(temp ** 2) ** 0.5
            # temp= np.abs( batch_y_n - batch_y )[batch_M==1]
            # loss_c[i_c//strd, 1] = np.mean(temp**2)**0.5

            # dk_aux.save_tensor_thumbs(batch_x, batch_y, batch_y_n, y_s, True, i_c, i_eval, thumbs_dir )

        # print('test cost   %.3f' % cost_c.mean())
        print(
            'Train loss   %.6f,   %.6f' % (loss_c[:i_c // strd + 1, 0].mean(), loss_c[:i_c // strd + 1, 1].mean()))

        strd = 2
        loss_c = np.zeros((X_test.shape[0] // strd + 1, 2))
        start_point = np.random.randint(strd)

        for i_c in range(start_point, X_test.shape[0], strd):

            y_s = np.zeros((SX, SY, SZ, n_tar))
            y_c = np.zeros((SX, SY, SZ))

            for lx in lx_list:
                for ly in ly_list:
                    for lz in lz_list:

                        condition2 = True  # np.sum(mask_test[i_c:i_c + 1, lx:lx + LX, ly:ly + LY, lz:lz + LZ] == 0) < \
                        # (1 - prop_nonzero_patch) * LX * LY * LZ

                        if condition2 and np.min(M_test[i_c:i_c + 1, lx + LXc:lx + LX - LXc, ly + LYc:ly + LY - LYc,
                                                 lz + LZc:lz + LZ - LZc]) > 0:
                            batch_x = X_test[i_c:i_c + 1, lx:lx + LX, ly:ly + LY, lz:lz + LZ, :].copy()

                            pred_temp = sess.run(Y_pred, feed_dict={X: batch_x, p_keep_conv: keep_test})

                            y_s[lx:lx + LX, ly:ly + LY, lz:lz + LZ, :] += pred_temp[0, :, :, :, :]
                            y_c[lx:lx + LX, ly:ly + LY, lz:lz + LZ] += 1

            y_s = y_s / np.tile(y_c[:, :, :, np.newaxis] + 0.00001, [1, 1, 1, n_tar])

            # batch_x = X_test[i_c, :, :, :, :].copy()
            batch_y = Y_test[i_c, :, :, :, :].copy()
            # batch_y_n = Y_base_test[i_c, :, :, :, :].copy()
            batch_M = M_test[i_c, :, :, :].copy()

            temp = np.abs(y_s - batch_y)[batch_M == 1]
            loss_c[i_c // strd, 0] = np.mean(temp ** 2) ** 0.5
            # temp= np.abs( batch_y_n - batch_y )[batch_M==1]
            # loss_c[i_c//strd, 1] = np.mean(temp**2)**0.5

        print('Test loss   %.6f,   %.6f' % (loss_c[:i_c // strd + 1, 0].mean(), loss_c[:i_c // strd + 1, 1].mean()))

        if loss_c[:i_c + 1, 0].mean() < best_test:
            # dk_aux.empty_folder(thumbs_dir)
            time.sleep(5)
            print('Saving new model checkpoint.')
            best_test = loss_c[:i_c + 1, 0].mean()
            temp_path = thumbs_dir + 'model_' + "%03d" % i_eval + '_' + str(
                int(round(10000.0 * loss_c[:i_c + 1, 0].mean()))) + '.ckpt'
            saver.save(sess, temp_path)

        if i_eval == 0:
            loss_old = loss_c[:i_c + 1, 0].mean()
        else:
            if loss_c[:i_c + 1, 0].mean() > 1.0 * loss_old:
                L_Rate = L_Rate * 0.90
                epochs_w_no_progress += 1
            else:
                epochs_w_no_progress = 0
            loss_old = loss_c[:i_c + 1, 0].mean()

        print('learning rate and mean test dice:  ', L_Rate, loss_old, epochs_w_no_progress)

print("Training time in hours ", (time.time() - start_time) / 3600)