import glob
import os
import threading

import nibabel as nib
import numpy as np
from dipy.reconst.shm import sf_to_sh
from dipy.core.sphere import Sphere


def process_directory(directory, index, X_train, Y_train, Mask, directory_list, x_begin, x_end, y_begin, y_end, z_begin,
                      z_end, x_name='dwi_6_1000_sh.nii.gz', y_name="wm.nii.gz", b0_name=None, bvec_name=None,
                      sh_order=2):
    print('Processing directory {} of {}: {}'.format(
        index + 1, len(directory_list), directory))
    # Load the nifti file
    x_data = nib.load(os.path.join(directory, x_name)).dataobj[x_begin:x_end, y_begin:y_end, z_begin:z_end, :]
    if b0_name is not None:
        b0_data = nib.load(os.path.join(directory, b0_name)).dataobj[x_begin:x_end, y_begin:y_end, z_begin:z_end, :]
        # non-zero divide x by b0
        # print("min and max before divide: ", x_data.min(), x_data.max())
        x_data = np.divide(x_data, b0_data, out=np.zeros_like(x_data), where=b0_data != 0)
        # print("min and max after divide: ", x_data.min(), x_data.max())
        del b0_data

    if bvec_name is not None:
        bvecs = np.loadtxt(os.path.join(directory, bvec_name)).T
        sphere_bvecs = Sphere(xyz=np.asarray(bvecs))
        x_data = sf_to_sh(x_data, sphere=sphere_bvecs, sh_order=sh_order, basis_type='tournier07')

    y_data = nib.load(os.path.join(directory, y_name)).dataobj[x_begin:x_end, y_begin:y_end, z_begin:z_end, :]
    mask = nib.load(os.path.join(directory, 'mask.nii.gz')).get_fdata().astype(bool)
    mask = mask[..., 0] if mask.ndim == 4 else mask

    # x_data = x_data[x_begin:x_end, y_begin:y_end, z_begin:z_end, :]
    # y_data = y_data[x_begin:x_end, y_begin:y_end, z_begin:z_end, :]
    mask = mask[x_begin:x_end, y_begin:y_end, z_begin:z_end]

    # # Flatten the middle three dimensions
    # x_data = x_data.reshape((-1, x_data.shape[-1]))
    # y_data = y_data.reshape((-1, y_data.shape[-1]))
    # mask = mask.reshape((-1, mask.shape[-1]))

    X_train[index] = x_data
    Y_train[index] = y_data
    Mask[index] = mask

    del x_data, y_data, mask


def process_data(data_dir, dir_list_dir, target_shape=(119, 138, 96), x_begin=16, x_end=135, y_begin=3, y_end=141,
                 z_begin=0, z_end=96, dwi_name='dwi_6_1000_sh.nii.gz', y_name="wm.nii.gz", b0_name=None, N_grad=6,
                 bvec_name=None, sh_order=2):
    try:
        directory_list = np.loadtxt(dir_list_dir, dtype=str)
        directory_list = [os.path.join(data_dir, directory)
                          for directory in directory_list]
        print('Loaded directory list from file.')

    except OSError:
        print('Could not load directory list from file. Generating new list...')
        print('If you do not want to generate a new list, please terminate the program within 5 seconds.')
        os.system('sleep 5')
        # Generate all folders containing `d_mri_subset_6.nii.gz`
        directory_list = glob.glob(
            os.path.join('.', '*', dwi_name))
        directory_list = [os.path.dirname(directory)
                          for directory in directory_list]
        # np.random.shuffle(directory_list)

        np.savetxt(os.path.join('.', 'directory_list.txt'),
                   directory_list, fmt='%s')

    print('Found {} directories containing `{}`'.format(
        len(directory_list), dwi_name))
    print(directory_list)

    n_sig = N_grad  # 15 #because of SH4, otherwise: N_grad
    n_tar = 45  # SH6

    sx, sy, sz = target_shape

    X_train = np.empty((len(directory_list), sx, sy, sz, n_sig))
    Y_train = np.empty((len(directory_list), sx, sy, sz, n_tar))
    Mask = np.empty((len(directory_list), sx, sy, sz), dtype=bool)

    threads = []
    for i, directory in enumerate(directory_list):
        thread = threading.Thread(target=process_directory, args=(
            directory, i, X_train, Y_train, Mask, directory_list, x_begin, x_end, y_begin, y_end, z_begin, z_end,
            dwi_name, y_name, b0_name, bvec_name, sh_order))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    print(X_train.shape)
    print(X_train.min(), X_train.max())
    print(Y_train.shape)
    print(Y_train.min(), Y_train.max())
    print(Mask.shape)

    # mask as uint
    Mask = Mask.astype(bool)

    return X_train, Y_train, Mask
