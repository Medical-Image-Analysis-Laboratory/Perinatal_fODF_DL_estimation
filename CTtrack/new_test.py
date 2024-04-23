import threading
import numpy as np
import os
from tqdm import tqdm
import nibabel as nib
from dipy.reconst.shm import sf_to_sh
from dipy.core.sphere import Sphere
from utils.utils import *
from utils.model import *


class NewTest:
    def __init__(self, n_sig, n_tar, model_dir,
                 data_dir, data_list, data_file_name, bvec_file_name, mask_file_name,
                 output_dir=None, mul_coe=1, output_file_name='wm.nii.gz', sh_order=4, batch_size=2000):
        self.n_sig = n_sig
        self.n_tar = n_tar
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.data_list = data_list
        self.data_file_name = data_file_name
        self.mask_file_name = mask_file_name
        self.bvec_file_name = bvec_file_name
        self.output_dir = output_dir if output_dir is not None else data_dir
        self.mul_coe = mul_coe
        self.output_file_name = output_file_name
        self.sh_order = sh_order
        self.batch_size = batch_size

    def load_model(self):
        self.model = network(self.n_sig)
        self.model.load_weights(self.model_dir)

    def predict_batch(self, dwi, fod, xyz):
        """
        dwi: (batch_size, 3, 3, 3, n_sig)
        fod: (H, W, D, n_tar)
        xyz: (batch_size, 3)
        """
        # y = self.model.predict(dwi, verbose=0)
        y = self.model(dwi, training=False)
        fod[xyz[:, 0], xyz[:, 1], xyz[:, 2], :] = y

    def test(self):
        self.load_model()

        if isinstance(self.data_list, str):
            self.data_list = np.loadtxt(self.data_list, dtype=str)
        self.data_list = [os.path.join(self.data_dir, d)
                          for d in self.data_list]

        for dir in tqdm(self.data_list):
            dwi_nii = nib.load(os.path.join(dir, self.data_file_name))
            dwi = dwi_nii.get_fdata()

            if self.bvec_file_name is not None:
                bvecs = np.loadtxt(os.path.join(dir, self.bvec_file_name)).T
                dwi = np.clip(dwi, 0, 1)
                sphere_bvecs = Sphere(xyz=bvecs)
                dwi = sf_to_sh(dwi, sphere=sphere_bvecs,
                               sh_order=self.sh_order, basis_type='tournier07')

            mask = nib.load(os.path.join(dir, self.mask_file_name)
                            ).get_fdata().astype(bool)
            mask = mask[..., 0] if mask.ndim == 4 else mask

            xyz = np.argwhere(mask)
            fod = np.zeros((*mask.shape, self.n_tar))
            dwi_in = np.zeros((len(xyz), 3, 3, 3, self.n_sig))

            for i, (x, y, z) in enumerate(xyz):
                x_slice = slice(max(x - 1, 0), min(x + 2, dwi.shape[0]))
                y_slice = slice(max(y - 1, 0), min(y + 2, dwi.shape[1]))
                z_slice = slice(max(z - 1, 0), min(z + 2, dwi.shape[2]))

                dwi_in[i] = np.pad(dwi[x_slice, y_slice, z_slice], [(0, 3 - x_slice.stop + x_slice.start), (0,
                                                                    3 - y_slice.stop + y_slice.start), (0, 3 - z_slice.stop + z_slice.start), (0, 0)], mode='constant')
                
            fod_out = self.model.predict(dwi_in, verbose=0, batch_size=self.batch_size, workers=16, use_multiprocessing=True)

            fod[xyz[:, 0], xyz[:, 1], xyz[:, 2], :] = fod_out
            # threads = []
            # for i in range(0, len(xyz), self.batch_size):
            #     batch_xyz = xyz[i:i+self.batch_size] if i + \
            #         self.batch_size < len(xyz) else xyz[i:]
            #     batch_dwi = np.zeros((len(batch_xyz), 3, 3, 3, self.n_sig))
            #     for j, (x, y, z) in enumerate(batch_xyz):
            #         x_slice = slice(max(x - 1, 0), min(x + 2, dwi.shape[0]))
            #         y_slice = slice(max(y - 1, 0), min(y + 2, dwi.shape[1]))
            #         z_slice = slice(max(z - 1, 0), min(z + 2, dwi.shape[2]))

            #         batch_dwi[j] = np.pad(dwi[x_slice, y_slice, z_slice], [(0, 3 - x_slice.stop + x_slice.start), (0,
            #                               3 - y_slice.stop + y_slice.start), (0, 3 - z_slice.stop + z_slice.start), (0, 0)], mode='constant')

            #     t = threading.Thread(
            #         target=self.predict_batch, args=(batch_dwi, fod, batch_xyz))
            #     t.start()
            #     threads.append(t)

            # for t in threads:
            #     t.join()

            # fod /= self.mul_coe
            fod_nii = nib.Nifti1Image(fod, dwi_nii.affine)
            nib.save(fod_nii, os.path.join(self.output_dir,
                     os.path.basename(dir), self.output_file_name))

            del fod_nii, fod, dwi_nii, dwi, mask, xyz, dwi_in, fod_out


if __name__ == '__main__':
    n_sig = 6
    n_tar = 45
    model_dir = '/media/rizhong/Elements/images_340_shard/CTtrack.hdf5'

    data_dir = '/media/rizhong/Elements/images_340_shard/'
    data_list = np.loadtxt(
        '/media/rizhong/Elements/images_340_shard/subjects.txt', dtype=str)[219+13+3:]
    data_file_name = 'd_mri_subset_6.nii.gz'
    mask_file_name = 'mask_orig.nii.gz'
    bvec_file_name = 'b_vecs_selected.bvec'
    output_dir = '/media/rizhong/Elements/images_340_shard/'
    mul_coe = 1
    output_file_name = 'dl_CTtrack.nii.gz'
    sh_order = 2
    batch_size = 2000

    # # Use CPU only
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    test = NewTest(n_sig, n_tar,  model_dir,
                   data_dir, data_list, data_file_name, bvec_file_name, mask_file_name,
                   output_dir, mul_coe, output_file_name, sh_order, batch_size)
    test.test()

    del test
