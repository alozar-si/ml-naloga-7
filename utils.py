import numpy as np
from tensorflow.keras import backend as K

def sampling(args, latent_dim=2):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                                mean=0.0, stddev=1.0)
    
    return z_mean + K.exp(0.5 * z_log_var) * epsilon