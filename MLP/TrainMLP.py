import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import numpy as np
import os
import datetime
from tqdm import tqdm
from os.path import join

import crl_aux
import MLP.dk_model as dk_model
from MLP.make_h5 import process_data


# Define directories for output, train, models, and thumbs
output_dir = join('/home/ch051094/Desktop', 'TrainMLP')
train_dir = join(output_dir, 'trains', 'train_03')
model_dir = join(train_dir, 'models')

# Create directories if they do not exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Load and reshape the data
X_train, Y_train, Mask_train, X_val, Y_val, Mask_val = process_data()

X_train = X_train.reshape(-1, X_train.shape[-1])
Y_train = Y_train.reshape(-1, Y_train.shape[-1])
Mask_train = Mask_train.flatten()
X_val = X_val.reshape(-1, X_val.shape[-1])
Y_val = Y_val.reshape(-1, Y_val.shape[-1])
Mask_val = Mask_val.flatten()

print(f'Data loaded at {datetime.datetime.now()}')

# Get training data dimensions
n_train, n_sig = X_train.shape[0], X_train.shape[-1]
n_val, n_tar = Y_val.shape[0], Y_val.shape[-1]

# Set GPU index and learning rate
gpu_ind = 0
L_Rate = 1e-2

# Construct n_feat_vec, where n_sig is 6 and n_tar is 45
n_feat_vec = [n_sig, 40, 40, 40, 50, 60, 70, n_tar]
# n_feat_vec = [n_sig, 300, 300, 300, 400, 500, 600, n_tar]

# Initialize placeholders
X = tf.placeholder(tf.float32, [None, n_sig])
Y = tf.placeholder(tf.float32, [None, n_tar])
learning_rate = tf.placeholder(tf.float32)
p_keep_hidden = tf.placeholder(tf.float32)

# Define model, cost function, and optimizer
Y_p_un = dk_model.davood_reg_net_SH(X, n_feat_vec, p_keep_hidden, bias_init=0.001)
Y_s = tf.reduce_sum(Y_p_un, axis=1)
Y_p = tf.math.divide_no_nan(Y_p_un, tf.reshape(Y_s, [-1, 1]))
cost = tf.reduce_mean(tf.pow((Y - Y_p), 2))
adam_optimizer = tf.train.AdamOptimizer(learning_rate)
optimizer = adam_optimizer.minimize(cost)

# Define CUDA environment variables
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ind)

# Initialize saver and session
saver = tf.train.Saver(max_to_keep=50)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Initialize variables for training loop
i_global = 0
best_test = 0
i_eval = -1
batch_size = 2000
n_epochs = 10000
test_interval = n_train // batch_size * 5


# Define a function to calculate cost
def calculate_cost(n_samples, X_samples, Y_samples, batch_size, cost, sess, X, Y, p_keep_hidden):
    cost_v = np.zeros(n_samples // batch_size)
    pbar = tqdm(total=n_samples // batch_size, ascii=True)
    for i_v in range(n_samples // batch_size):
        batch_x = X_samples[i_v * batch_size:(i_v + 1) * batch_size, :].copy()
        batch_y = Y_samples[i_v * batch_size:(i_v + 1) * batch_size, :].copy()
        cost_v[i_v] = sess.run(cost, feed_dict={X: batch_x, Y: batch_y, p_keep_hidden: 1.0})
        pbar.update(1)
    pbar.close()
    return cost_v.mean()


# Initialize minimum validation cost as infinity
min_val_cost = float('inf')
best_model_path = None

print(f'Training starts at {datetime.datetime.now()}')

# Number of all voxels
n_voxels = Mask_train.shape[0]
# Desired proportion of non-zero voxels in a batch
prop_nonzero_patch = 0.75

# Get indices of in-mask and out-mask voxels
in_mask_indices = np.where(Mask_train != 0)[0]
out_mask_indices = np.where(Mask_train == 0)[0]

# Calculate the number of in-mask and out-mask voxels in each batch
n_in_mask_batch = int(batch_size * 0.75)  # 75% of batch size
n_out_mask_batch = batch_size - n_in_mask_batch  # The remaining 25%

# Calculate the number of batches per epoch
n_batches_per_epoch = n_train // batch_size

for epoch_i in range(n_epochs):
    # Precompute random batches for this epoch
    in_mask_batches = np.random.choice(in_mask_indices, (n_batches_per_epoch, n_in_mask_batch), replace=True)
    out_mask_batches = np.random.choice(out_mask_indices, (n_batches_per_epoch, n_out_mask_batch), replace=True)

    for i_train in tqdm(range(n_train // batch_size), ascii=True):
        # Get the next precomputed batch
        in_mask_q = in_mask_batches[i_train]
        out_mask_q = out_mask_batches[i_train]

        # Combine the selected voxels
        q = np.concatenate([in_mask_q, out_mask_q])

        # Get the corresponding X and Y batches
        batch_x = X_train[q, :].copy()
        batch_y = Y_train[q, :].copy()

        # Run the optimizer
        sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y, learning_rate: L_Rate, p_keep_hidden: 1.0})
        i_global += 1

    print('\nEpoch: ', epoch_i, ',  Global Step: ', i_global,)

    cost_v = calculate_cost(n_train, X_train, Y_train, batch_size, cost, sess, X, Y, p_keep_hidden)
    print('Train cost  %.6f' % (cost_v * 1000))

    if i_global % test_interval == 0:
        i_eval += 1

        val_cost = calculate_cost(n_val, X_val, Y_val, batch_size, cost, sess, X, Y, p_keep_hidden)
        print('Test cost   %.6f' % (val_cost * 1000))

        # Save the model if it is the best one so far
        if val_cost < min_val_cost:
            min_val_cost = val_cost
            temp_path = join(model_dir, f'model_saved_best_epoch_{epoch_i}_val_cost_{val_cost * 1000:.6f}.ckpt')
            saver.save(sess, temp_path)
            best_model_path = temp_path  # Save the path of the best model

# Restore the best model
saver.restore(sess, best_model_path)