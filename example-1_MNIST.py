#%%
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
import numpy as np
from utils import sampling

(x_train, y_train), _ = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_train = x_train.reshape((len(x_train)), np.prod(x_train.shape[1:]))
#%%
# Implementacija enkoderja:
original_dim = np.prod(x_train.shape[1:]) # dimenzija vhodnih podatkov
hidden_dim = 64 # skriti sloj z 64 node -i
latent_dim = 2 # 2D latentni prostor

# Input layer
inputs = keras.Input(shape=(original_dim ,) )
# First hidden layer
h = keras.layers.Dense(hidden_dim, activation ='selu')(inputs)

z_mean = keras.layers.Dense(latent_dim)(h)
z_log_var = keras.layers.Dense(latent_dim)(h)

# Layer that calculates value z = z_mean + z_var * epsilon
z = keras.layers.Lambda(sampling)([z_mean, z_log_var])

#Define model for encoder:
# encoder returns z_mean, z_log_var, z
encoder = keras.Model(inputs, [z_mean, z_log_var, z], name='encoder')
# %%
# Implementation of decoder
latent_inputs = keras.Input(shape=(latent_dim, ), name='z_sampling')
# hidden layer x
x = keras.layers.Dense(hidden_dim, activation='selu')(latent_inputs)
outputs = keras.layers.Dense(original_dim, activation='sigmoid')(x)
decoder = keras.Model(latent_inputs, outputs, name='decoder')
# %%
# Define VAE
outputs = decoder(encoder(inputs)[2])
vae = keras.Model(inputs, outputs, name='vae')

rec_loss = keras.losses.binary_crossentropy(inputs, outputs)
rec_loss *= original_dim
kl_loss = -0.5*K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(rec_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer="adam")
# %%
# Train model on train data
batch_size = 32
history = vae.fit(x_train, x_train, epochs=50, batch_size=batch_size, validation_data=None)
# %%
