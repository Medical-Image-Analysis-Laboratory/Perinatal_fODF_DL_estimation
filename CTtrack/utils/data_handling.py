import numpy as np
import os
import glob
import nibabel as nib
from dipy.data import get_sphere
from dipy.core.sphere import Sphere, HemiSphere
from dipy.reconst.shm import sph_harm_lookup, smooth_pinv
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel
from dipy.reconst.dti import fractional_anisotropy
import dipy.reconst.dti as dti  
from dipy.segment.mask import median_otsu


class DataHandler(object):

    def __init__(self, params, mode):

        self.params = params
        self.dwi_path = self.params.data
        self.brain_mask_path = self.params.bm
        self.wm_mask_path = self.params.wm

        assert mode in ['train', 'track'], 'mode must be either train or track.'
        if mode == 'train':
            self.labels_path = self.params.labels
        else:
            self.labels_path = None
       
        self.dwi = None
        self.bvals = None
        self.bvecs = None
        self.brain_mask = np.array([])
        self.wm_mask = np.array([])
    
        if self.dwi_path is not None:
            self.load_dwi()
            self.load_b_table()

        if self.brain_mask_path is not None:
            self.brain_mask = self.load_mask(self.brain_mask_path)
        else:
            self.brain_mask = self.get_bm()
            
        if self.wm_mask_path is not None:
            self.wm_mask = self.load_mask(self.wm_mask_path)
        else:
            self.wm_mask = self.get_wm_mask()
            
        if self.labels_path is not None:
            self.load_labels()

    def load_dwi(self):
        dwi_file = get_file_path(os.getcwd(), self.dwi_path, "*.nii*")
        dwi_data = nib.load(dwi_file)
        self.dwi = dwi_data.get_data().astype("float32")
        self.affine = dwi_data.affine
        del dwi_data
        
    def load_labels(self):
        data = nib.load(self.labels_path)
        self.labels = data.get_data().astype("float32")

    def load_b_table(self):
        bval_file = get_file_path(os.getcwd(), self.dwi_path, "*.bvals")
        bvec_file = get_file_path(os.getcwd(), self.dwi_path, "*.bvecs")
        self.bvals, self.bvecs = read_bvals_bvecs(bval_file, bvec_file)
        self.gtab = gradient_table(self.bvals, self.bvecs)
        
    def get_wm_mask(self):
        tenmodel = TensorModel(self.gtab)
        tenfit = tenmodel.fit(self.dwi, mask=self.brain_mask)
        FA = fractional_anisotropy(tenfit.evals)
        MD = dti.mean_diffusivity(tenfit.evals)
        wm_mask = (np.logical_or(FA >= 0.25, (np.logical_and(FA >= 0.12, MD >= 0.001))))
        return wm_mask
        
    def get_bm(self):
        b0_mask, brain_mask = median_otsu(self.dwi, median_radius=1, numpass=2, vol_idx=[0,0])
        return brain_mask


    # The folowing code for normalizing, computing SH, and resampling are taken from DeepTract (https://github.com/itaybenou/DeepTract).
    @staticmethod
    def normalize_dwi(weights, b0):
        b0 = b0[..., None] 
        nb_erroneous_voxels = np.sum(weights > b0)
        if nb_erroneous_voxels != 0:
            weights = np.minimum(weights, b0)

        # Normalize dwi using the b0.
        weights_normed = weights / b0
        weights_normed[np.logical_not(np.isfinite(weights_normed))] = 0.

        return weights_normed

    def get_spherical_harmonics_coefficients(self, dwi_weights, bvals, bvecs, sh_order=8, smooth=0.006):
        # Exract the averaged b0.
        b0_idx = bvals == 0
        b0 = dwi_weights[..., b0_idx].mean(axis=3) + 1e-10

        # Extract diffusion weights and normalize by the b0.
        bvecs = bvecs[np.logical_not(b0_idx)]
        weights = dwi_weights[..., np.logical_not(b0_idx)]
        weights = self.normalize_dwi(weights, b0)

        raw_sphere = Sphere(xyz=bvecs)

        # Fit SH to signal
        sph_harm_basis = sph_harm_lookup.get("tournier07")
        Ba, m, n = sph_harm_basis(sh_order, raw_sphere.theta, raw_sphere.phi)
        L = -n * (n + 1)
        invB = smooth_pinv(Ba, np.sqrt(smooth) * L)
        data_sh = np.dot(weights, invB.T)
        return data_sh

    def resample_dwi(self, directions=None, sh_order=10, smooth=0.006):
        print('initializing')
        sphere = get_sphere('repulsion100')
        if directions is not None:
            sphere = Sphere(xyz=directions)

        sph_harm_basis = sph_harm_lookup.get("tournier07")
        Ba, m, n = sph_harm_basis(sh_order, sphere.theta, sphere.phi)
        self.Ba = Ba
        
        self.b0_idx = self.bvals == 0
        self.b0 = self.dwi[..., self.b0_idx].mean(axis=3) + 1e-10

        bvecs = self.bvecs[np.logical_not(self.b0_idx)]
        raw_sphere = Sphere(xyz=bvecs)

        # Fit SH to signal
        sph_harm_basis = sph_harm_lookup.get("tournier07")
        Ba, m, n = sph_harm_basis(sh_order, raw_sphere.theta, raw_sphere.phi)
        L = -n * (n + 1)
        self.invB = smooth_pinv(Ba, np.sqrt(smooth) * L)
      
       
    
    def preprocess(self, X, b0):
        weights = X[..., np.logical_not(self.b0_idx)]
        weights_normed = self.normalize_dwi(weights, b0)
        data_sh = np.dot(weights_normed, self.invB.T)
        data_resampled = np.dot(data_sh, self.Ba.T)
        return data_resampled

    def mask_dwi(self):
        dwi_vol = self.dwi.shape
        mask_vol = self.brain_mask
        self.dwi = self.dwi * np.tile(mask_vol[..., None], (1, 1, 1, dwi_vol[-1]))
       
        
    @staticmethod
    def load_mask(mask_path):
        dwi_data = nib.load(mask_path)
        return dwi_data.get_data().astype("float32")
        
def get_range(idx, size):
    lower_b = idx-1 if idx-1>=0 else 0
    upper_b = idx+2 if idx+2<=size else idx+1 if idx+1<=size else idx
    return lower_b, upper_b
    
    

def prepare_labels(labels, num_outputs):
    return labels.ravel()
    
    

def get_file_path(curr_path, target_dir, extension):
    os.chdir(target_dir)
    for file in glob.glob(extension):
        file_path = os.path.join(target_dir, file)
    os.chdir(curr_path)

    return file_path
