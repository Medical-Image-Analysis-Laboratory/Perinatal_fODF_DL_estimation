#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 08:25:14 2022

@author: ch209389
"""
import os
from os import listdir
from os.path import isdir, join
import numpy as np
from shutil import copyfile
import nibabel as nib
import pandas as pd
import time
import dipy.core.sphere as dipysphere
import SimpleITK as sitk
import crl_aux
import dk_seg

###############################################################################


#   Generate the training data
dhcp_dir = '/home/ch051094/Desktop/dhcp_4_hamza/'
subj_info = pd.read_csv(dhcp_dir + 'participants.tsv', delimiter='\t')

# to save the images that will be used for model training
base_dir = '/home/ch051094/Desktop/DataDWI2FOD/'
images_dir = base_dir + 'images/'
training_dir = base_dir + 'DL_90sub_EqualSampling9-TestSet15-RandomBvecsCondNumber/'
mapInputToSH = True

os.makedirs(training_dir, exist_ok=True)
prefix = ''  # '/home/ch051094/anaconda3/bin/' #to be added for right mrtrix path

n_max_subjects = 100  # 73
subj_count = 0

min_age = 25
max_age = 50  # 38

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

N_grad_1 = 6
Xp, Yp, Zp = crl_aux.optimized_six()
sphere_grad = dipysphere.Sphere(Xp, Yp, Zp)
v_grad_1, _ = sphere_grad.vertices, sphere_grad.faces

N_grad_list = [N_grad_1]
V_grad_list = [v_grad_1]

temp = []
start_time = time.time()

for subj_ind in range(n_max_subjects):
    subj = subj_info['participant_id'][subj_ind]
    print('Processing subject ', subj)

    #anat_dir= dhcp_dir + 'dhcp_anat_pipeline/sub-' + subj
    dmri_dir = dhcp_dir + 'dhcp_dmri_pipeline/sub-' + subj

    dmri_sess = [d for d in listdir(dmri_dir) if isdir(join(dmri_dir, d))]

    sess_tsv =  dmri_dir + '/sub-' + subj + '_sessions.tsv' #anat_dir + '/sub-' + subj + '_sessions.tsv'
    sess_info= pd.read_csv( sess_tsv , delimiter= '\t')


    if len(dmri_sess)==0 or not sess_info.loc[0, 'session_id']== int(dmri_sess[0].split('ses-')[1]):

        continue

    else:

        age_c= sess_info.loc[0, 'scan_age']

        if age_c<min_age or age_c>max_age:

            continue

        else:

            if subj_count < n_max_subjects:

                subj_count+= 1

                print('Reading data for Subject ' + str(subj_ind) + '.')

                subject= 'sub-' + subj
                session= 'ses-'+ str(sess_info.loc[0, 'session_id'])

                dwi_dir = dmri_dir + '/' + session + '/' + 'dwi/'
                #ant_dir = anat_dir + '/' + session + '/' + 'anat/'

                subject_results_dir= images_dir + subj+'/'  #+ '_test/'
                os.makedirs(subject_results_dir)


                os.chdir(subject_results_dir)


                # copy the main diffusion volume
                file_name= subject + '_' + session + '_desc-preproc_dwi.nii.gz'
                copyfile( dwi_dir + file_name , subject_results_dir + 'dwi.nii.gz' )

                # denoising
                # my_command= 'dwidenoise dwi.nii.gz dwi_denoised.nii.gz -noise noise.nii.gz -quiet'
                # os.system(my_command)


                # read the b values/vectors and copy them into the subject directory as well
                file_name= subject + '_' + session + '_desc-preproc_dwi'
                b_vals= np.loadtxt( dwi_dir + file_name + '.bval' )
                b_vecs= np.loadtxt( dwi_dir + file_name + '.bvec' )
                copyfile( dwi_dir + file_name + '.bval', subject_results_dir + 'bvals.bval' )
                copyfile( dwi_dir + file_name + '.bvec', subject_results_dir + 'bvecs.bvec' )

                file_name= subject + '_' + session + '_desc-preproc_space-dwi_brainmask.nii.gz'
                copyfile( dwi_dir + file_name , subject_results_dir + 'mask_orig.nii.gz' )



                # bias correction
                #os.system('FSLDIR=/opt/el7/pkgs/fsl/fsl-5.0.11/bin/fsl:$FSLDIR')
                #os.system('export FSLDIR')
                my_command= 'dwibiascorrect fsl dwi.nii.gz dwi_debiased.nii.gz  -fslgrad bvecs.bvec bvals.bval  -quiet'
                my_command+= ' -mask  mask_orig.nii.gz '
                my_command+= ' -bias  estimated_bias.nii.gz '
                os.system(my_command)

                os.remove(subject_results_dir + 'dwi.nii.gz')



                # read the denoised volume
                dmri_volume= sitk.ReadImage( subject_results_dir + 'dwi_debiased.nii.gz')
                dmri_volume= sitk.GetArrayFromImage(dmri_volume)
                dmri_volume= np.transpose( dmri_volume, [3,2,1,0] )



                # interpolate the data

                brain_mask_img= sitk.ReadImage( subject_results_dir + 'mask_orig.nii.gz' )
                original_spacing= brain_mask_img.GetSpacing()
                original_size= brain_mask_img.GetSize()

                n_dwi= dmri_volume.shape[-1]
                SX, SY, SZ= [int(original_size[i]/desired_spacing[i]*original_spacing[i]) for i in range(3)]

                d_img0= dmri_volume.copy()
                dmri_volume= np.zeros( (SX, SY, SZ,n_dwi), np.float32 )
                for i in range(n_dwi):
                    temp= d_img0[:,:,:,i].copy()
                    temp= np.transpose( temp, [2,1,0] )
                    temp= sitk.GetImageFromArray(temp)
                    temp.SetDirection( brain_mask_img.GetDirection() )
                    temp.SetOrigin( brain_mask_img.GetOrigin() )
                    temp.SetSpacing( brain_mask_img.GetSpacing() )
                    temp= dk_seg.resample3d(temp, original_spacing, desired_spacing, sitk.sitkBSpline)
                    temp= sitk.GetArrayFromImage(temp)
                    temp= np.transpose( temp, [2,1,0] )
                    dmri_volume[:,:,:,i]= temp
                del d_img0



                # Read and resample the brain mask
                brain_mask_img= dk_seg.resample3d(brain_mask_img, original_spacing, desired_spacing, sitk.sitkNearestNeighbor)
                brain_mask= sitk.GetArrayFromImage(brain_mask_img)
                brain_mask= np.transpose( brain_mask, [2,1,0] )


                # the directiona, origin, and spacig of the mask, which may be needed below
                ref_dir= brain_mask_img.GetDirection()
                ref_org= brain_mask_img.GetOrigin()
                ref_spc= brain_mask_img.GetSpacing()



                # Compute an "eroded" mask to remove some of the boundary voxels
                skull_eroded= crl_aux.skull_from_brain_mask(brain_mask, radius= 2.0)
                b0_img= dmri_volume[:,:,:,b_vals==0]
                b0_img= np.mean(b0_img, axis=-1)
                compute_mask= np.logical_and( brain_mask==1, b0_img>1)
                compute_mask_eroded= np.logical_and( skull_eroded==0, compute_mask>0)
                compute_mask_img= np.transpose(compute_mask, [2,1,0])
                compute_mask_img= sitk.GetImageFromArray(compute_mask_img.astype(np.float64))
                compute_mask_img.SetDirection(ref_dir)
                compute_mask_img.SetOrigin(ref_org)
                compute_mask_img.SetSpacing(ref_spc)
                sitk.WriteImage(compute_mask_img, subject_results_dir + 'compute_mask_img.nii.gz' )
                compute_mask_eroded_img= np.transpose(compute_mask_eroded, [2,1,0])
                compute_mask_eroded_img= sitk.GetImageFromArray(compute_mask_eroded_img.astype(np.float64))
                compute_mask_eroded_img.SetDirection(ref_dir)
                compute_mask_eroded_img.SetOrigin(ref_org)
                compute_mask_eroded_img.SetSpacing(ref_spc)
                sitk.WriteImage(compute_mask_eroded_img, subject_results_dir + 'compute_mask_eroded_img.nii.gz' )



                #
                # # seelct subsets of the data for diffusion tensor estimation and fiber orientation estimation
                # ind_0_1000= np.where( np.logical_or(b_vals==0, b_vals==1000)  )[0]
                # b_vals_0_1000= b_vals[ind_0_1000]
                # b_vecs_0_1000= b_vecs[:,ind_0_1000]
                # gtab_0_1000 = gradient_table(b_vals_0_1000, b_vecs_0_1000)
                # d_img_0_1000= dmri_volume[:,:,:,ind_0_1000]
                #
                # # ind_0_2600= np.where( np.logical_or(b_vals==0, b_vals==2600)  )[0]
                # # b_vals_0_2600= b_vals[ind_0_2600]
                # # b_vecs_0_2600= b_vecs[:,ind_0_2600]
                # # gtab_0_2600 = gradient_table(b_vals_0_2600, b_vecs_0_2600)
                # # d_img_0_2600= dmri_volume[:,:,:,ind_0_1000]
                #
                #
                temp= nib.load( subject_results_dir + 'compute_mask_img.nii.gz' )
                temp.header['srow_x']= temp.affine[0,:]
                temp.header['srow_y']= temp.affine[1,:]
                temp.header['srow_z']= temp.affine[2,:]
                affine= temp.affine
                #
                # ###################################################################
                #
                #
                # # DTI
                #
                # np.savetxt(subject_results_dir + 'b_vecs_0_1000.txt', b_vecs_0_1000, delimiter=' ')
                # np.savetxt(subject_results_dir + 'b_vals_0_1000.txt', b_vals_0_1000, delimiter=' ')
                # os.rename('b_vecs_0_1000.txt', 'b_0_1000.bvec')
                # os.rename('b_vals_0_1000.txt', 'b_0_1000.bval')
                #
                # array_img = nib.Nifti1Image(d_img_0_1000, affine)
                # nib.save(array_img, subject_results_dir + 'd_img_0_1000.nii.gz' )
                #
                # my_command= prefix+'dwi2tensor d_img_0_1000.nii.gz tensor_mrtrix.nii.gz' \
                #             + ' -mask compute_mask_img.nii.gz -fslgrad b_0_1000.bvec b_0_1000.bval -quiet'
                # os.system(my_command)
                #
                # my_command= prefix+'tensor2metric tensor_mrtrix.nii.gz' \
                #             + ' -adc  MD_mrtrix.nii.gz  -fa   FA_mrtrix.nii.gz -quiet'
                # os.system(my_command)
                #
                # os.remove(subject_results_dir + 'd_img_0_1000.nii.gz')
                # os.remove(subject_results_dir + 'b_0_1000.bvec')
                # os.remove(subject_results_dir + 'b_0_1000.bval')

                #MSMT-CSD to generate groud-truth FOD

                temp = nib.Nifti1Image(dmri_volume, affine)
                nib.save(temp, subject_results_dir + 'd_img.nii.gz' )

                my_command= prefix+'dwi2response dhollander d_img.nii.gz -fslgrad bvecs.bvec bvals.bval ' + \
                            ' -mask compute_mask_eroded_img.nii.gz  response_wm.txt response_gm.txt response_csf.txt -quiet'
                os.system(my_command)

                my_command= prefix+'dwi2fod msmt_csd   d_img.nii.gz  -fslgrad bvecs.bvec bvals.bval  '
                my_command+= ' -mask  compute_mask_img.nii.gz '
                my_command+= ' response_wm.txt wm.nii.gz '
                my_command+= ' response_gm.txt gm.nii.gz  '
                my_command+= ' response_csf.txt csf.nii.gz -quiet'
                os.system(my_command)

                os.remove(subject_results_dir + 'd_img.nii.gz')
                os.remove(subject_results_dir + 'response_wm.txt')
                os.remove(subject_results_dir + 'response_gm.txt')
                os.remove(subject_results_dir + 'response_csf.txt')

                # Select data subset MULTISHELL

                b_vals_all = [400, 1000, 2600]
                l=-1
                for N_grad, v_grad in zip(N_grad_list, V_grad_list):
                    l=l+1
                    #print('N_grad', N_grad)
                    #print('b_vals_all[l]', b_vals_all[l])
                    b_vecs_to_chose_from = b_vecs[:, b_vals == b_vals_all[l]]

                    grad_ind_norot, grad_ang_norot, _ = crl_aux.find_closest_bvecs_no_rotation(v_grad.T, \
                                                                                               b_vecs_to_chose_from.T,
                                                                                               antipodal=True)
                    #Useful for code of half GS
                    ##print('grad_ind_norot',grad_ind_norot)
                    #indices_to_chose_from = range(N_grad*2)
                    ##print('b_vecs_to_chose_from', b_vecs_to_chose_from)
                    #grad_ind_norot = np.setdiff1d(indices_to_chose_from,grad_ind_norot).astype(int)
                    ##print('grad_ind_norot', grad_ind_norot)

                    if not len(np.unique(grad_ind_norot)) == N_grad:

                        print('Non-unique directions selected')
                        print(N_grad-len(np.unique(grad_ind_norot)) )
                    #else:

                    print('mean and std of angles  %.3f ' % grad_ang_norot.mean(),
                          '  %.3f ' % grad_ang_norot.std(), )

                    DWI_dhcp_norot = dmri_volume[:, :, :, b_vals == b_vals_all[l]][:, :, :, grad_ind_norot].copy()

                    for i in range(N_grad):
                        temp = DWI_dhcp_norot[:, :, :, i] / b0_img
                        temp[compute_mask < 0.5] = 0
                        DWI_dhcp_norot[:, :, :, i] = temp.copy()

                    DWI_dhcp_norot[DWI_dhcp_norot < 0] = 0
                    DWI_dhcp_norot[DWI_dhcp_norot > 1] = 1
                    DWI_dhcp_norot = nib.Nifti1Image(DWI_dhcp_norot, affine)
                    nib.save(DWI_dhcp_norot, subject_results_dir + 'd_mri_subset_' + str(N_grad) + '.nii.gz')

                    b_vals_selected_curr = b_vals[b_vals == b_vals_all[l]][grad_ind_norot]
                    b_vecs_selected_curr = b_vecs[:, b_vals == b_vals_all[l]][:, grad_ind_norot]

                    if l==0:
                        b_vals_0 = b_vals[b_vals == 0][10:]
                        #print('b_vecs.shape', b_vecs.shape)
                        b_vecs_0 = b_vecs[:, b_vals == 0][:,10:]
                        #print('b_vecs_0.shape',b_vecs_0.shape)

                        b_vals_selected = np.hstack((b_vals_0, b_vals_selected_curr))
                        b_vecs_selected = np.hstack((b_vecs_0, b_vecs_selected_curr))
                    else:
                        b_vals_selected = np.hstack((b_vals_selected, b_vals_selected_curr))
                        b_vecs_selected = np.hstack((b_vecs_selected, b_vecs_selected_curr))

                    if l==0:
                        d_img_0 = dmri_volume[:, :, :, b_vals == 0][...,10:]
                        d_img_selected_curr = dmri_volume[:, :, :, b_vals == b_vals_all[l]][:, :, :, grad_ind_norot]
                        d_img_selected = np.concatenate((d_img_0, d_img_selected_curr), axis=-1)
                    else:
                        d_img_selected_curr = dmri_volume[:, :, :, b_vals == b_vals_all[l]][:, :, :, grad_ind_norot]
                        #print('d_img_selected_curr.shape ',d_img_selected_curr.shape)
                        d_img_selected = np.concatenate((d_img_selected, d_img_selected_curr), axis=-1)

                    np.savetxt(subject_results_dir + 'b_vals_selected.txt', b_vals_selected_curr, delimiter=' ')
                    np.savetxt(subject_results_dir + 'b_vecs_selected.txt', b_vecs_selected_curr, delimiter=' ')
                    os.rename('b_vals_selected.txt', 'b_vals_selected.bval')
                    os.rename('b_vecs_selected.txt', 'b_vecs_selected.bvec')

                    d_img_selected_nib = nib.Nifti1Image(d_img_selected, affine)
                    nib.save(d_img_selected_nib, subject_results_dir + 'd_img_selected.nii.gz')

                my_command = prefix + 'dwi2response dhollander d_img2_selected.nii.gz -fslgrad b_vecs2_selected.bvec b_vals2_selected.bval ' + \
                             ' -mask compute_mask_eroded_img.nii.gz response_wm.txt response_gm.txt response_csf.txt -quiet'
                os.system(my_command)

                my_command = prefix + 'dwi2fod msmt_csd d_img2_selected.nii.gz -fslgrad b_vecs2_selected.bvec b_vals2_selected.bval  '
                my_command+= ' -mask  compute_mask_img.nii.gz '
                my_command+= ' response_wm.txt wm.nii.gz '
                my_command+= ' response_gm.txt gm.nii.gz  '
                my_command+= ' response_csf.txt csf.nii.gz -quiet'

                os.system(my_command)

                os.remove(subject_results_dir + 'd_img2_selected.nii.gz')
                os.remove(subject_results_dir + 'response_wm.txt')
                os.remove(subject_results_dir + 'response_gm.txt')
                os.remove(subject_results_dir + 'response_csf.txt')

                os.remove(subject_results_dir + 'b_vals_selected.bval')
                os.remove(subject_results_dir + 'b_vecs_selected.bvec')

                # Apply CSD on all target_bval volumes or on data subset

                target_bval = 2600
                for N_grad, v_grad in zip(N_grad_list, V_grad_list):

                    DWI_dhcp_norot = dmri_volume[:,:,:,b_vals == target_bval] #[:,:,:,grad_ind_norot].copy()

                    for i in range(N_grad):
                        temp= DWI_dhcp_norot[:,:,:,i]/b0_img
                        temp[compute_mask<0.5]= 0
                        DWI_dhcp_norot[:,:,:,i]= temp.copy()

                    DWI_dhcp_norot[DWI_dhcp_norot<0]= 0
                    DWI_dhcp_norot[DWI_dhcp_norot>1]= 1
                    DWI_dhcp_norot = nib.Nifti1Image(DWI_dhcp_norot, affine)
                    nib.save(DWI_dhcp_norot, subject_results_dir + 'd_mri_subset_' +  str(N_grad) + '.nii.gz' )

                    b_vals_0= b_vals[b_vals==0]
                    b_vecs_0= b_vecs[:,b_vals==0]
                    b_vals_selected= b_vals[b_vals==target_bval]#[grad_ind_norot]
                    b_vecs_selected= b_vecs[:,b_vals==target_bval]#[:,grad_ind_norot]
                    b_vals_selected= np.hstack((b_vals_0,b_vals_selected))
                    b_vecs_selected= np.hstack((b_vecs_0,b_vecs_selected))

                    d_img_0= dmri_volume[:,:,:,b_vals==0]
                    d_img_selected= dmri_volume[:,:,:,b_vals==target_bval]#[:,:,:,grad_ind_norot]
                    d_img_selected= np.concatenate((d_img_0,d_img_selected), axis=-1)

                    np.savetxt(subject_results_dir + 'b_vals_selected.txt', b_vals_selected, delimiter=' ')
                    np.savetxt(subject_results_dir + 'b_vecs_selected.txt', b_vecs_selected, delimiter=' ')
                    os.rename('b_vals_selected.txt', 'b_vals_selected.bval')
                    os.rename('b_vecs_selected.txt', 'b_vecs_selected.bvec')

                    d_img_selected = nib.Nifti1Image(d_img_selected, affine)
                    nib.save(d_img_selected, subject_results_dir + 'd_img_selected.nii.gz' )

                    my_command= prefix+'dwi2response tournier d_img_selected.nii.gz -fslgrad b_vecs_selected.bvec b_vals_selected.bval ' + \
                                ' -mask compute_mask_eroded_img.nii.gz response_csd.txt -quiet'
                    os.system(my_command)

                    my_command= prefix+'dwi2fod csd d_img_selected.nii.gz response_csd.txt csd_fodf_mrtrix_1000_' \
                                +  str(N_grad) + '.nii.gz  -mask  compute_mask_img.nii.gz '\
                                +    ' -fslgrad b_vecs_selected.bvec b_vals_selected.bval -quiet'
                    os.system(my_command)

                    os.remove(subject_results_dir + 'd_img_selected.nii.gz')
                    os.remove(subject_results_dir + 'response_csd.txt')
                    os.remove(subject_results_dir + 'b_vals_selected.bval')
                    os.remove(subject_results_dir + 'b_vecs_selected.bvec')


print("Data generation took (in hours) ", (time.time()-start_time)/3600)

