import numpy as np
from dipy.data import get_sphere, default_sphere
from dipy.core.sphere import Sphere, HemiSphere
from keras import backend as K
from dipy.core.geometry import sphere_distance
from dipy.reconst.shm import sph_harm_lookup, smooth_pinv
from utils.data_handling import *
import tensorflow as tf
import threading
import numpy as np
from scipy.special import rel_entr

def get_indices(shape):
    indices = np.zeros([*shape[0:3], 3])
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                indices[i,j,k] = [i,j,k]
    return indices.reshape(-1, 3).astype(int)

def JSD():
    def jsd_calc(y_true, y_pred):
        y_true = tf.math.abs(y_true)
        y_pred = tf.math.abs(y_pred)
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        y_true /= K.sum(y_true, axis=-1, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        y_true = K.clip(y_true, K.epsilon(), 1 - K.epsilon())
        
        kl_12 = y_true * K.log(y_true/y_pred) 
        kl_12 = tf.where(tf.math.is_nan(kl_12), tf.zeros_like(kl_12), kl_12)
        kl_12 = -K.sum(kl_12, -1)
        
        kl_21 = y_pred * K.log(y_pred/y_true) 
        kl_21 = tf.where(tf.math.is_nan(kl_21), tf.zeros_like(kl_21), kl_21)
        kl_21 = -K.sum(kl_21, -1)
        
        jsd = tf.math.abs(0.5*(kl_12+kl_21))
        return jsd
    return jsd_calc


def soft_f1():
    def macro_soft_f1(y, y_hat):
        y = tf.cast(y, tf.float32)
        y_hat = tf.cast(y_hat, tf.float32)
        tp = tf.reduce_sum(y_hat * y, axis=0)
        fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
        fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
        soft_f1 = 2*tp / (2*tp + fn + fp + 1e-16)
        cost = 1 - soft_f1 # reduce 1 - soft-f1 in order to increase soft-f1
        macro_cost = tf.reduce_mean(cost) # average on all labels
        return macro_cost
    return macro_soft_f1


# def get_range(idx, size):
#     lower_b = idx-1 if idx-1>=0 else 0
#     upper_b = idx+2 if idx+2<=size else idx+1 if idx+1<=size else idx
#     return lower_b, upper_b
    
    

# def prepare_labels(labels, num_outputs):
#     return labels.ravel()

   

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
            b0 = np.ones([3, 3, 3])
            the_mask = np.zeros([3, 3, 3])
            vicinity = X[lx:ux, ly:uy, lz:uz]
            block[lx-i+1: ux-i+1,  ly-j+1:uy-j+1, lz-k+1:uz-k+1] = vicinity
            b0[lx-i+1: ux-i+1,  ly-j+1:uy-j+1, lz-k+1:uz-k+1] = b_zero[lx:ux, ly:uy, lz:uz]
            label = prepare_labels(y[i,j,k], output_size)
            the_mask[lx-i+1: ux-i+1,  ly-j+1:uy-j+1, lz-k+1:uz-k+1] = mask[lx:ux, ly:uy, lz:uz]
            block = block * np.tile(the_mask[..., None], (1, 1, 1, X.shape[-1]))
            label = label * the_mask[1,1,1]
            
            X_batch.append(block)
            y_batch.append(label)
            b0_batch.append(b0)
            
            is_over = (index == len(train_index)-1)
            
            if len(X_batch) == batch_size or is_over:
                processed_batch = data_handler.preprocess(np.asarray(X_batch), np.asarray(b0_batch))
                X_batch = np.asarray(processed_batch)
                y_batch = np.asarray(y_batch)
                X_batch_padded = np.zeros([batch_size, *processed_batch.shape[1:]])
                X_batch_padded[:len(X_batch)] = X_batch
                y_batch_padded = np.zeros([batch_size, *label.shape])
                y_batch_padded[:len(X_batch)] = y_batch

                yield X_batch, y_batch
                X_batch = []
                y_batch = []
                b0_batch = []
                        
