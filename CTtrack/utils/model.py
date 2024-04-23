# 5
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Dropout
import numpy as np

# sinusoidal position encoding
def get_3d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    grid_h = np.arange(grid_size)
    grid_w = np.arange(grid_size)
    grid_d = np.arange(grid_size)
    i,j,k = np.meshgrid(grid_w, grid_h, grid_d, indexing='ij') 

    grid = np.stack([
    np.reshape(i, [-1]),
    np.reshape(j, [-1]),
    np.reshape(k, [-1]),
    ])

    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[0]) 
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[1])  
    emb_D = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[2])  
    emb = np.concatenate([emb_h, emb_w, emb_D], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

class Patches(layers.Layer):
    def __init__(self, num_patches):
        super(Patches, self).__init__()
        self.num_patches = num_patches

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
        })
        return config

    def call(self, input_data):
        patch_dims = input_data.shape[-1]
        patches = tf.reshape(input_data, [-1, self.num_patches, patch_dims])
        return patches

class PatchEncoder(layers.Layer):
    def __init__(self, proj_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = 27
        self.positional_embed_dim = 6
        self.projection_dim = proj_dim
        self.projection = layers.Dense(units=self.projection_dim - self.positional_embed_dim)       
        self.position_embedding = layers.Embedding(
            input_dim=self.num_patches, output_dim=self.positional_embed_dim,input_shape=(self.num_patches,3),
        )
        rows = tf.range(0, 3, delta=1)
        cols = tf.range(0, 3, delta=1)
        slices = tf.range(0, 3, delta=1)

        i,j,k = tf.meshgrid(cols, rows, slices, indexing='ij')
        indices = tf.stack([
        tf.reshape(i, [-1]),
        tf.reshape(j, [-1]),
        tf.reshape(k, [-1]),
        ])
        self.positions = tf.transpose(indices)

        # Use this for sinusoidal position encoding
        # self.position_embedding = get_3d_sincos_pos_embed(embed_dim=self.positional_embed_dim, grid_size=3, cls_token=False)
        

    def call(self, patch):
        pe = tf.reduce_mean(self.position_embedding(self.positions), axis=1)
        pe = tf.repeat(tf.expand_dims(pe, axis=0), len(patch), 0)   
        patch_embedding = self.projection(patch) 
        encoded = tf.concat([patch_embedding, tf.cast(pe, tf.float32)], axis=2)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({
            "projection_dim": self.projection_dim,
        })
        return config




def residual_block(x, filter, kernel_size = (1,1,1)):
    
    x = layers.LayerNormalization()(x)
    x_skip = x
    x_skip = layers.Conv3D(filter, (1,1,1), padding = 'same')(x_skip)
   
    x = layers.Conv3D(filter, kernel_size, padding = 'same')(x)
    x = layers.Conv3D(4*filter, (1,1,1), padding = 'same')(x)
    x = layers.Activation(tf.nn.gelu)(x)
    x = layers.Conv3D(filter, (1,1,1), padding = 'same')(x)

    # Add Residue
    x = layers.Add()([x, x_skip]) 
    return x


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


def transformer_block(x, projection_dim, num_heads, transformer_units, dropout):
    x1 = layers.LayerNormalization(epsilon=1e-6)(x)
    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=projection_dim, dropout=dropout)(x1, x1)
    x2 = layers.Add()([attention_output, x])
    x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
    x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=dropout)
    x = layers.Add()([x3, x2])
    return x
    


def network(grad_directions, projection_dim=128, transformer_layers=4, transformer_units=[4*128, 128],\
    num_heads=4, mlp_head_units=[1024], num_output=45, dropout=0.02):

    # 3D CNN projector
    inputs = keras.Input(shape=(3,3,3,grad_directions,), name='diffusion_data')
    encod = residual_block(inputs, 60, kernel_size = (1,1,1))
    projector =  keras.Model(inputs = inputs, outputs = encod, name="projector")
    
    
    # Transformer
    patches = Patches(num_patches=27)(encod)
    encoded_patches = PatchEncoder(projection_dim)(patches)
    for _ in range(transformer_layers):
        encoded_patches = transformer_block(encoded_patches, projection_dim, num_heads, transformer_units, dropout)

    attention_map = layers.LayerNormalization(epsilon=1e-6,  name='attention_map')(encoded_patches)
    transformer = keras.Model(inputs = encod, outputs = attention_map, name="transformer")   


    # Regressor head
    representation = layers.GlobalAvgPool1D()(attention_map) 
    x = mlp(representation, mlp_head_units, dropout_rate=0.05)
    pred = layers.Dense(num_output)(x)
    regressor = keras.Model(inputs = attention_map, outputs = pred, name="mlp_head")

    encoded = projector(inputs)
    features = transformer(encoded)
    out = regressor(features)
    model = keras.Model(inputs = inputs, outputs = out, name="model")
    
    return model


if __name__ == '__main__':
    model = network(15)
    model.summary()
    print(model.input_shape, model.output_shape)

    x = np.random.random((40, 3, 3, 3, 15))
    y = model(x)
    print(y.shape)
