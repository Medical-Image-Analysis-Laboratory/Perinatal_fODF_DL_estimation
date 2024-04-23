from utils.model import *
from utils.utils import *
from tensorflow.keras.callbacks import  ModelCheckpoint, CSVLogger, TensorBoard
from sklearn.model_selection import train_test_split
from keras.models import load_model
from os.path import join
import os
import time
import numpy as np
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
import wandb

from data import process_data




class Train(object):

    def __init__(self, args):
        super().__init__()
        self.params = args
        print(self.params.labels, os.path.exists(self.params.labels))
        self.model = None
        self.output_size = 45
        self.learning_rate = self.params.learning_rate
        self.batch_size = self.params.train_batch_size
        self.epochs = self.params.epochs
        self.split_ratio = self.params.split_ratio
        self.dropout_prob = self.params.dropout_prob
        self.model_save_dir = self.params.save_dir
        self.trained_model_dir = self.params.trained_model_dir
        if self.trained_model_dir is None:
            self.model_file = join(self.model_save_dir, f'CTtrack-SS3T-{self.params.grad_directions}.h5')
        else: 
            self.model_file = self.trained_model_dir

        # self.data_handler = DataHandler(self.params, mode='train')


    def set_model(self, grad_directions):
        self.model = network(grad_directions)
        return

    def train(self):
        model_trained_on = f'CTtrack-SS3T-{self.params.grad_directions}'
        wandb.init(project='FODNet', entity='tobiasforest',
                name=model_trained_on, save_code=True, mode='online')
        wandb.run.log_code(os.getcwd())

        # Set data
        print('preprocessing...')
        t0 = time.time()
        
        target_shape = (74, 87, 64)
        x_begin, x_end, y_begin, y_end, z_begin, z_end = 14, 88, 3, 90, 0, 64

        train_data, train_labels, train_mask = process_data(
            data_dir='/media/rizhong/Data/dHCP_train',
            dir_list_dir='/home/rizhong/Documents/FODNet/train_val_list.txt',
            target_shape=target_shape,
            x_begin=x_begin,
            x_end=x_end,
            y_begin=y_begin,
            y_end=y_end,
            z_begin=z_begin,
            z_end=z_end,
            dwi_name=f'dwi_{self.params.grad_directions}_1000_orig.nii.gz',
            y_name='wm_ss3t_20_0_88_1000.nii.gz',
            b0_name='b0.nii.gz',
            N_grad=self.params.grad_directions,
            bvec_name=f'dwi_{self.params.grad_directions}_1000.bvec',
            sh_order=self.params.sh_order,
        )

        print("train_data.shape: ", train_data.shape)
        print("train_labels.shape: ", train_labels.shape)
        print("train_mask.shape: ", train_mask.shape)

        # train_mask: np.array [145, 80, 94, 72]
        # now, we need to get the indices of the voxels whose mask is True
        # indices: np.array [N, 4]; N: number of voxels
        self.indices = np.argwhere(train_mask)
        print("indices.shape: ", self.indices.shape)

        grad_directions = self.params.grad_directions
        # self.indices = get_indices(data_handler.dwi.shape)
        train_index, valid_index,_,_ = train_test_split(self.indices, np.arange(len(self.indices)), test_size=1-self.split_ratio)
        print(f'done\t{time.time()-t0}s\n')
        
        # Set model
        self.set_model(grad_directions)
        if os.path.exists(self.model_file):
            print('loading model...')
            self.model.load_weights(self.model_file)
        
        self.model.summary()
            
        print(self.model.input_shape, self.model.output_shape)
        print()
    
            
        # Set optimizer
        step = tf.Variable(0, trainable=False)
        schedule = tf.optimizers.schedules.PiecewiseConstantDecay(
            [100, 200], [1e-2, 1e-3, 1e-4])
        lr = 1e-2 * schedule(step)

        wd =  0.5e-2
        # optimizer = tfa.optimizers.AdamW(
        # learning_rate=lr, weight_decay=wd)
        # optimizer = keras.optimizers.Adam(learning_rate=lr, weight_decay=wd)
        # use AdamW instead of Adam
        optimizer = tf.keras.optimizers.experimental.AdamW(
            learning_rate=lr, weight_decay=wd)
        
        loss = keras.losses.MeanAbsoluteError()
        metric = keras.metrics.MeanSquaredError()
        
        self.model.compile(loss=loss, optimizer=optimizer, metrics = metric)

        callbacks = []
        callbacks.append(WandbModelCheckpoint(monitor='val_loss',
                         filepath=join(self.model_file),
                         save_best_only=True,
                         # save_weights_only=True,
                         verbose=1,
                         mode="min",
                         ))
        

        # callbacks.append(TensorBoard(log_dir=join(self.model_save_dir, 'Tensorboard_logs'),
        #                      update_freq = 500))
        # callbacks.append(CSVLogger(join(self.model_save_dir, 'training.log'), append=True, separator=';'))
        callbacks.append(WandbMetricsLogger(log_freq='batch'))
    
        train_history = self.model.fit(
                my_generator(train_index, train_data, train_labels, self.batch_size, num_directions=grad_directions),
                epochs=self.epochs,
                verbose=2,
                callbacks=callbacks,
                steps_per_epoch=np.ceil(float(len(train_index))) / float(self.batch_size),
                validation_data=my_generator(valid_index, train_data, train_labels, self.batch_size, num_directions=grad_directions),
                validation_steps=np.ceil(float(len(valid_index)) / float(self.batch_size)),
                shuffle=True,
                workers=48,
                use_multiprocessing=True,
        )
     
        return
    




