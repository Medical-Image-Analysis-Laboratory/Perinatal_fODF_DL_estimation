import glob
import os
import threading

import h5py
import nibabel as nib
import numpy as np


def process_directory(directory, index, X_train, Y_train, Mask, directory_list, x_begin, x_end, y_begin, y_end, z_begin,
                      z_end):
    print('Processing directory {} of {}: {}'.format(index + 1, len(directory_list), directory))
    # Load the nifti file
    x_data = nib.load(os.path.join(directory, 'd_mri_subset_6_sh.nii.gz')).get_fdata()
    y_data = nib.load(os.path.join(directory, 'wm.nii.gz')).get_fdata()
    mask = nib.load(os.path.join(directory, 'compute_mask_eroded_img.nii.gz')).get_fdata()

    x_data = x_data[x_begin:x_end, y_begin:y_end, z_begin:z_end, :]
    y_data = y_data[x_begin:x_end, y_begin:y_end, z_begin:z_end, :]
    mask = mask[x_begin:x_end, y_begin:y_end, z_begin:z_end]

    # # Flatten the middle three dimensions
    # x_data = x_data.reshape((-1, x_data.shape[-1]))
    # y_data = y_data.reshape((-1, y_data.shape[-1]))
    # mask = mask.reshape((-1, mask.shape[-1]))

    X_train[index] = x_data
    Y_train[index] = y_data
    Mask[index] = mask


def process_data(target_shape=(119, 138, 96), x_begin=16, x_end=135, y_begin=3, y_end=141, z_begin=0, z_end=96):
    try:
        directory_list = np.loadtxt(os.path.join('/home/ch051094/Desktop', 'directory_list.txt'), dtype=str)
        print('Loaded directory list from file.')

    except OSError:
        # Generate all folders containing `d_mri_subset_6.nii.gz`
        directory_list = glob.glob(
            os.path.join('/home/ch051094/Desktop/DataDWI2FOD/images', '*', 'd_mri_subset_6_sh.nii.gz'))
        directory_list = [os.path.dirname(directory) for directory in directory_list]
        np.random.shuffle(directory_list)

        np.savetxt(os.path.join('/home/ch051094/Desktop', 'directory_list.txt'), directory_list, fmt='%s')

    print('Found {} directories containing `d_mri_subset_6_sh.nii.gz`'.format(len(directory_list)))
    print(directory_list)

    N_grad = 6

    n_sig = N_grad  # 15 #because of SH4, otherwise: N_grad
    n_tar = 45  # SH6

    sx, sy, sz = target_shape

    X_train = np.zeros((len(directory_list), sx, sy, sz, n_sig))
    Y_train = np.zeros((len(directory_list), sx, sy, sz, n_tar))
    Mask = np.zeros((len(directory_list), sx, sy, sz))

    # return X_train[:70], Y_train[:70], X_train[70:], Y_train[70:]

    threads = []
    for i, directory in enumerate(directory_list):
        thread = threading.Thread(target=process_directory, args=(
            directory, i, X_train, Y_train, Mask, directory_list, x_begin, x_end, y_begin, y_end, z_begin, z_end))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    print(X_train.shape)
    print(X_train.min(), X_train.max())
    print(Y_train.shape)
    print(Y_train.min(), Y_train.max())
    print(Mask.shape)

    return X_train[:70], Y_train[:70], Mask[:70], X_train[70:], Y_train[70:], Mask[70:]
