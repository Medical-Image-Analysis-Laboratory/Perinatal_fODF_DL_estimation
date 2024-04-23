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
import threading


class DataHandler(object):

    def __init__(self, params, mode):

        self.params = params
        self.dwi_path = self.params.data
        self.brain_mask_path = self.params.bm
        self.wm_mask_path = self.params.wm

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
        self.load_b_table()
        
        b0 = self.bvals <= 5
        single_shell_bval = 1000
        b_single = np.logical_and(self.bvals<=single_shell_bval+5, self.bvals>=single_shell_bval-5)
        ind_0_single= np.logical_or(b0, b_single)
        self.dwi =  self.dwi[:,:,:,ind_0_single]
        self.bvecs= self.bvecs[ind_0_single,:]
        self.bvals= self.bvals[ind_0_single]
        self.gtab = gradient_table(self.bvals, self.bvecs)
        print(f'Number of single shell directions: {sum(b_single)}')
        
    def load_labels(self):
        data = nib.load(self.labels_path)
        self.labels =  data.get_data().astype("float32")

    def load_b_table(self):
        bval_file = get_file_path(os.getcwd(), self.dwi_path, "*.bvals")
        bvec_file = get_file_path(os.getcwd(), self.dwi_path, "*.bvecs")
        self.bvals, self.bvecs = read_bvals_bvecs(bval_file, bvec_file)
        
        
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


    def resample_dwi(self, directions=200, sh_order=8, smooth=0.002):
        print(f'resampling diffusion data on {directions} directions...')
        # sphere = get_sphere('repulsion200')
        Xp, Yp, Zp= distribute_on_hemisphere(directions)
        sphere = Sphere(Xp, Yp, Zp)
        # self.v = sphere.vertices

        sph_harm_basis = sph_harm_lookup.get("tournier07")
        # descoteaux07
        Ba, m, n = sph_harm_basis(sh_order, sphere.theta, sphere.phi)
        self.Ba = Ba
        
        self.b0_idx = self.bvals == np.min(self.bvals)
        self.b0 = self.dwi[:,:,:,self.b0_idx].mean(axis=3) + 1e-6
        self.bvecs = self.bvecs[np.logical_not(self.b0_idx)]
        
        raw_sphere = Sphere(xyz=self.bvecs)
        Ba, m, n = sph_harm_basis(sh_order, raw_sphere.theta, raw_sphere.phi)
        L = -n * (n + 1)
        self.invB = smooth_pinv(Ba, np.sqrt(smooth) * L)
        
        self.dwi = self.dwi[..., np.logical_not(self.b0_idx)]
        nb_erroneous_voxels = np.sum(self.dwi > self.b0[..., None])
        if nb_erroneous_voxels != 0:
            self.dwi = np.minimum(self.dwi, self.b0[..., None])
        self.dwi /= self.b0[..., None]  

        self.mean_val = np.mean(self.dwi)
    
    
    
    def preprocess(self, X):
        data_sh = np.dot(X, self.invB.T)
        data_resampled = np.dot(data_sh, self.Ba.T)
        # print(f'max is: {np.max(data_resampled)}, min is :{np.min(data_resampled)}')
        return data_resampled

    def mask_data(self):
        dwi_vol = self.dwi.shape
        mask_vol = self.brain_mask
        self.dwi = self.dwi * np.tile(mask_vol[..., None], (1, 1, 1, dwi_vol[-1]))
        if self.labels_path is not None:
            labels_vol = self.labels.shape
            self.labels = self.labels * np.tile(mask_vol[..., None], (1, 1, 1, labels_vol[-1]))



        
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
    
    
def distribute_on_hemisphere(n):
    indices = np.arange(0, n, dtype=float) + 0.5
    phi = np.arccos(1 - indices/n)
    theta = np.pi * (1 + 5**0.5) * indices
    xp, yp, zp = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi);
    return xp, yp, zp

    

def get_file_path(curr_path, target_dir, extension):
    os.chdir(target_dir)
    for file in glob.glob(extension):
        file_path = os.path.join(target_dir, file)
    os.chdir(curr_path)

    return file_path

def get_indices(shape):
    indices = np.zeros([*shape[0:3], 3])
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                indices[i,j,k] = [i,j,k]
    return indices.reshape(-1, 3).astype(int)





class ThreadSafeIterator:

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    def g(*args, **kwargs):
        return ThreadSafeIterator(f(*args, **kwargs))
    return g


@threadsafe_generator

def generator(train_index, data_handler, output_size, batch_size):
    X = data_handler.dwi
    y = data_handler.labels
    noise_std = data_handler.mean_val * np.arange(0.02,0.22,0.01)
    b_zero = data_handler.b0
    mask = data_handler.brain_mask
    while True:
        X_batch = []
        y_batch = []
        b0_batch = []
        for index in range(len(train_index)):
            i,j,k = train_index[index][0], train_index[index][1], train_index[index][2]
            lx, ux = get_range(i, X.shape[0])
            ly, uy = get_range(j, X.shape[1])
            lz, uz = get_range(k, X.shape[2])
            
            block = np.zeros([3, 3, 3, X.shape[-1]])
            vicinity = X[lx:ux, ly:uy, lz:uz]
            block[lx-i+1: ux-i+1,  ly-j+1:uy-j+1, lz-k+1:uz-k+1] = vicinity
            block += np.random.normal(0, np.random.choice(noise_std), size=block.shape)
            label = prepare_labels(y[i,j,k], output_size)
            
            X_batch.append(block)
            y_batch.append(label)
            
            is_over = (index == len(train_index)-1)
            
            if len(X_batch) == batch_size or is_over:
                processed_batch = data_handler.preprocess(np.asarray(X_batch))
                X_batch = np.asarray(processed_batch)
                y_batch = np.asarray(y_batch)
                X_batch_padded = np.zeros([batch_size, *processed_batch.shape[1:]])
                X_batch_padded[:len(X_batch)] = X_batch
                y_batch_padded = np.zeros([batch_size, *label.shape])
                y_batch_padded[:len(X_batch)] = y_batch

                # print(f'max: {np.max(X_batch_padded)}, min: {np.min(X_batch_padded)}')
                yield X_batch, y_batch
                X_batch = []
                y_batch = []
                b0_batch = []


@threadsafe_generator
def my_generator(indices, X, Y, batch_size, num_directions=6, num_sh=45):
    """
    indices: np.array [N, 4]; N: number of voxels
    X: np.array [145, H, W, D, 6]; 145: number of subjects; H, W, D: image size; 6: number of gradients
    Y: np.array [145, H, W, D, 45]; 45: number of SH coefficients
    batch_size: int

    yield: X_batch, Y_batch

    X_batch: np.array [batch_size, 3, 3, 3, 6]; 3, 3, 3: patch size; 6: number of gradients
    Y_batch: np.array [batch_size, 45]; 45: number of SH coefficients
    """
    while True:
        np.random.shuffle(indices)
        for i in range(0, len(indices), batch_size):
            # Adjust batch size for the last batch if necessary
            actual_batch_size = min(batch_size, len(indices) - i)
            batch_indices = indices[i:i + actual_batch_size]

            X_batch = np.empty((actual_batch_size, 3, 3, 3, num_directions))
            Y_batch = np.empty((actual_batch_size, num_sh))

            for j, index in enumerate(batch_indices):
                x0, y0, z0 = index[1], index[2], index[3]

                # Determine the slicing bounds, adjusting for edges
                x_slice = slice(max(x0 - 1, 0), min(x0 + 2, X.shape[1]))
                y_slice = slice(max(y0 - 1, 0), min(y0 + 2, X.shape[2]))
                z_slice = slice(max(z0 - 1, 0), min(z0 + 2, X.shape[3]))

                # Extract the patch and pad if necessary
                x_patch = X[index[0], x_slice, y_slice, z_slice]
                y_patch = Y[index[0], x0, y0, z0]

                X_batch[j] = np.pad(x_patch, [(0, 3 - x_patch.shape[0]), (0, 3 -
                                    x_patch.shape[1]), (0, 3 - x_patch.shape[2]), (0, 0)], mode='constant')
                Y_batch[j] = y_patch

            yield X_batch, Y_batch


if __name__ == '__main__':
    # test the my_generator
    # x = np.random.randint(0, 2, (20,60,60,60), dtype=bool)
    x = np.ones((20,60,60,60), dtype=bool)

    print(x.shape)

    indices = np.argwhere(x)

    print(indices.shape)

    X = np.random.random((20, 60, 60, 60, 6))
    Y = np.random.random((20, 60, 60, 60, 45))

    batch_size = 40

    gen = my_generator(indices, X, Y, batch_size)

    for i in range(10):
        X_batch, Y_batch = next(gen)
        print(X_batch.shape, Y_batch.shape)
        print(X_batch.dtype, Y_batch.dtype)
        print(X_batch.min(), X_batch.max())
        print(Y_batch.min(), Y_batch.max())
        print('-------------------')