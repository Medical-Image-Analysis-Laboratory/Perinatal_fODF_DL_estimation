#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 08:25:14 2022

@author: ch209389, ch235786
"""

import os
import numpy as np
import nibabel as nib
import pandas as pd
import time
import dipy.core.sphere as dipysphere
import crl_aux

###############################################################################

#   Generate the gold standard data (GS1 and GS2, halves of the ground truth)

dhcp_dir = '/fileserver/alborz/dhcp3/rel3_dhcp_dmri_shard_pipeline/'
subj_info = pd.read_csv(dhcp_dir + 'participants.tsv', delimiter='\t')

# to save the images that will be used for model training
base_dir = '/home/ch051094/Desktop/DataDWI2FOD_shard/'
images_dir = base_dir + 'images_340/'
mapInputToSH = True

if not os.path.exists(images_dir):
    os.makedirs(images_dir)

prefix = ''  # '/home/ch235786/anaconda3/bin/' #to be added for right mrtrix path

n_max_subjects = 10000  # 73
subj_count = 0

# number of measurement subsets

N_grad_1 = 32 #b400
Xp, Yp, Zp= crl_aux.distribute_on_hemisphere_spiral(N_grad_1)
sphere_grad = dipysphere.Sphere(Xp, Yp, Zp)
v_grad_1, _ = sphere_grad.vertices, sphere_grad.faces

N_grad_2 = 44 #b1000
Xp, Yp, Zp= crl_aux.distribute_on_hemisphere_spiral(N_grad_2)
sphere_grad = dipysphere.Sphere(Xp, Yp, Zp)
v_grad_2, _ = sphere_grad.vertices, sphere_grad.faces

N_grad_3 = 64 #b2600
Xp, Yp, Zp= crl_aux.distribute_on_hemisphere_spiral(N_grad_3)
sphere_grad = dipysphere.Sphere(Xp, Yp, Zp)
v_grad_3, _ = sphere_grad.vertices, sphere_grad.faces

N_grad_list= [ N_grad_1, N_grad_2, N_grad_3 ]
V_grad_list= [ v_grad_1, v_grad_2, v_grad_3 ]

start_time = time.time()

#for subj_ind in range(300,len(subj_info)):

for subj_ind in range(n_max_subjects):
    try:

        subj = subj_info['participant_id'][subj_ind]
        print('Processing subject ', subj)
        #anat_dir= dhcp_dir + 'dhcp_anat_pipeline/sub-' + subj
        dmri_dir = dhcp_dir + 'sub-' + subj
        if os.path.exists(images_dir+subj) and not os.path.exists(images_dir + subj+'/' + 'wmGS2.nii.gz'):

            if subj_count < n_max_subjects:

                subj_count+= 1

                print('Reading data for Subject ' + str(subj_ind) + '.')

                #ant_dir = anat_dir + '/' + session + '/' + 'anat/'

                subject_results_dir= images_dir + subj+'/'  #+ '_test/'
                if not os.path.exists(subject_results_dir):
                    os.makedirs(subject_results_dir)

                os.chdir(subject_results_dir)

                # Select data subset MULTISHELL

                b_vals= np.loadtxt( subject_results_dir + 'bvals.bval' )
                b_vecs= np.loadtxt( subject_results_dir+ 'bvecs.bvec' )
                dmri_volume = nib.load(subject_results_dir + 'dwi_debiased.nii.gz').get_data()
                compute_mask_img = nib.load( subject_results_dir + 'compute_mask_img.nii.gz' )
                affine= compute_mask_img.affine

                b_vals_all = [400, 1000, 2600]
                l=-1
                for N_grad, v_grad in zip(N_grad_list, V_grad_list):

                    l=l+1
                    #print('N_grad', N_grad)
                    #print('b_vals_all[l]', b_vals_all[l])
                    ind_target = np.logical_and(b_vals > b_vals_all[l]-100, b_vals < b_vals_all[l]+100)
                    b_vecs_to_chose_from = b_vecs[:, ind_target]
                    grad_ind_norot, grad_ang_norot, _ = crl_aux.find_closest_bvecs_no_rotation(v_grad.T, \
                                                                                               b_vecs_to_chose_from.T,
                                                                                               antipodal=True)
                    #Useful for code of half GS (run code twice)
                    #print('grad_ind_norot',grad_ind_norot)
                    indices_to_chose_from = range(N_grad*2)
                    ##print('b_vecs_to_chose_from', b_vecs_to_chose_from)
                    grad_ind_norot = np.setdiff1d(indices_to_chose_from,grad_ind_norot).astype(int)
                    ##print('grad_ind_norot', grad_ind_norot)

                    if not len(np.unique(grad_ind_norot)) == N_grad:

                        print('Non-unique directions selected')
                        print(N_grad-len(np.unique(grad_ind_norot)) )
                    #else:

                    print('mean and std of angles  %.3f ' % grad_ang_norot.mean(),
                          '  %.3f ' % grad_ang_norot.std(), )

                    b_vals_selected_curr = b_vals[ind_target][grad_ind_norot]
                    b_vecs_selected_curr = b_vecs[:, ind_target][:, grad_ind_norot]

                    if l==0:
                        b_vals_0 = b_vals[b_vals == 0][0:10]
                        #print('b_vecs.shape', b_vecs.shape)
                        b_vecs_0 = b_vecs[:, b_vals == 0][:,0:10]
                        #print('b_vecs_0.shape',b_vecs_0.shape)

                        b_vals_selected = np.hstack((b_vals_0, b_vals_selected_curr))
                        b_vecs_selected = np.hstack((b_vecs_0, b_vecs_selected_curr))
                    else:
                        b_vals_selected = np.hstack((b_vals_selected, b_vals_selected_curr))
                        b_vecs_selected = np.hstack((b_vecs_selected, b_vecs_selected_curr))

                    if l==0:
                        d_img_0 = dmri_volume[:, :, :, b_vals == 0][...,0:10]
                        d_img_selected_curr = dmri_volume[:, :, :, ind_target][:, :, :, grad_ind_norot]
                        d_img_selected = np.concatenate((d_img_0, d_img_selected_curr), axis=-1)
                    else:
                        d_img_selected_curr = dmri_volume[:, :, :, ind_target][:, :, :, grad_ind_norot]
                        #print('d_img_selected_curr.shape ',d_img_selected_curr.shape)
                        d_img_selected = np.concatenate((d_img_selected, d_img_selected_curr), axis=-1)

                    np.savetxt(subject_results_dir + 'b_vals_selectedGS2.txt', b_vals_selected, delimiter=' ')
                    np.savetxt(subject_results_dir + 'b_vecs_selectedGS2.txt', b_vecs_selected, delimiter=' ')
                    os.rename('b_vals_selectedGS2.txt', 'b_vals_selectedGS2.bval')
                    os.rename('b_vecs_selectedGS2.txt', 'b_vecs_selectedGS2.bvec')

                    d_img_selected_nib = nib.Nifti1Image(d_img_selected, affine)
                    nib.save(d_img_selected_nib, subject_results_dir + 'd_img_selectedGS2.nii.gz')

                my_command = prefix + 'dwi2response dhollander d_img_selectedGS2.nii.gz -fslgrad b_vecs_selectedGS2.bvec b_vals_selectedGS2.bval ' + \
                             ' -mask compute_mask_eroded_img.nii.gz response_wmGS2.txt response_gmGS2.txt response_csfGS2.txt -quiet -force'
                os.system(my_command)

                my_command = prefix + 'dwi2fod msmt_csd d_img_selectedGS2.nii.gz -fslgrad b_vecs_selectedGS2.bvec b_vals_selectedGS2.bval  '
                my_command+= ' -mask  compute_mask_img.nii.gz '
                my_command+= ' response_wmGS2.txt wmGS2.nii.gz '
                my_command+= ' response_gmGS2.txt gmGS2.nii.gz  '
                my_command+= ' response_csfGS2.txt csfGS2.nii.gz -quiet -force'

                os.system(my_command)

                os.remove(subject_results_dir + 'd_img_selectedGS2.nii.gz')
                os.remove(subject_results_dir + 'response_wmGS2.txt')
                os.remove(subject_results_dir + 'response_gmGS2.txt')
                os.remove(subject_results_dir + 'response_csfGS2.txt')

                os.remove(subject_results_dir + 'b_vals_selectedGS2.bval')
                os.remove(subject_results_dir + 'b_vecs_selectedGS2.bvec')
    except:
        print('Subject not processed ', subj_ind)


print("Data generation of gold standards (Delta GS) took (in hours) ", (time.time()-start_time)/3600)

















