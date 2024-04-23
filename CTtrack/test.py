from utils.utils import *
from utils.model import *
from os.path import join
import tensorflow as tf
from tqdm import tqdm
import time
import nibabel as nib
import subprocess
import os


class Test:
    def __init__(self, args):
        tf.config.experimental.set_visible_devices([], 'GPU')
        current_dir = os.getcwd()
        self.params = args
        self.save_dir = self.params.save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.track_batch_size = self.params.track_batch_size
        self.trained_model_dir = self.params.trained_model_dir
        self.model = None
        self.load_model()
        self.data_handler = DataHandler(self.params, mode='test')
        self.mask = self.data_handler.brain_mask
        self.wm_mask_path = self.data_handler.wm_mask_path
        self.brain_mask_path = self.data_handler.brain_mask_path
        if self.params.algorithm == 'deterministic':
            self.tractography_algorithm = 'SD_Stream'
        elif self.params.algorithm == 'probabilistic':
            self.tractography_algorithm = 'iFOD1'
        self.num_tracts = self.params.num_tracts
        self.max_length = self.params.max_length
        self.min_length = self.params.min_length

        
    def set_model(self):
        self.model = network(200)
        return

    def load_model(self):
        self.set_model()
        if os.path.exists(self.trained_model_dir):
            print('loading model...')
            self.model.load_weights(self.trained_model_dir)
        self.model.summary() 
        return
        
     
    def track(self):
       
        print('\nfinding SH...\n')
        # Set data'
        t0 = time.time()
        n_elem = 45
        self.data_handler.resample_dwi()
        print(f'done \t {round(time.time() - t0, 2)}s \n')
        
        X = self.data_handler.dwi
        self.data_handler.dwi = []
        prediction = np.zeros([*X.shape[0:3], n_elem])
        X_batch = []
        y_batch = []
        indices = []
        b0_batch = []
        for i in tqdm(range(X.shape[0])):
            for j in range(X.shape[1]):
                for k in range(X.shape[2]):
                    lx, ux = get_range(i, X.shape[0])
                    ly, uy = get_range(j, X.shape[1])
                    lz, uz = get_range(k, X.shape[2])
                    
                    block = np.zeros([3, 3, 3, X.shape[-1]])
                    vicinity = X[lx:ux, ly:uy, lz:uz]
                    block[lx-i+1: ux-i+1,  ly-j+1:uy-j+1, lz-k+1:uz-k+1] = vicinity
                    
                    X_batch.append(block)
                    indices.append([i,j,k])
                    is_over = (i==X.shape[0]-1 and j==X.shape[1]-1 and k==X.shape[2]-1)
                    
                    if len(X_batch) == self.track_batch_size or is_over:
                        processed_batch = self.data_handler.preprocess(np.asarray(X_batch))
                        X_batch = np.asarray(processed_batch)
                        X_batch_padded = np.zeros([self.track_batch_size, *processed_batch.shape[1:]])
                        X_batch_padded[:len(X_batch)] = X_batch
                        model_pred = self.model.predict(X_batch_padded)
                        pred = np.asarray(model_pred)[:len(X_batch)]
                        pred = pred.reshape(len(X_batch), n_elem)

                        idx = np.asarray(indices)
                        mask = self.mask[idx[:,0], idx[:,1], idx[:,2]]
                        pred = pred * mask[..., None]
                        prediction[idx[:,0], idx[:,1], idx[:,2],:] = pred
                        X_batch = []
                        indices = []

        
        prediction = np.asarray(prediction)
        prediction = prediction.reshape(*X.shape[0:3], -1)
        img = nib.Nifti1Image(prediction, self.data_handler.affine)
        nib.save(img, join(self.save_dir, 'SH.nii.gz')) 
        print('SH saved, performing tractography...\n')

        sh_dir = join(self.save_dir, 'SH.nii.gz')
        output_dir = join(self.save_dir, 'Tractogram.tck')
        subprocess.call(["tckgen -algorithm " + self.tractography_algorithm + " " +
                                            sh_dir + " " +
                                            output_dir +
                                            " -seed_image " + self.brain_mask_path +
                                            " -mask " + self.wm_mask_path +
                                            " -minlength " + str(self.min_length) + 
                                            " -maxlength " + str(self.max_length) + 
                                            " -select " + str(self.num_tracts)], shell=True)

        print(f'done \t {round(time.time() - t0, 2)}s \n')

