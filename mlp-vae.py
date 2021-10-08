import tensorflow as tf
from tensorflow.keras.layers import Lambda, Input, Dense, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.losses import mse, binary_crossentropy, mean_squared_error
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import h5py
import scipy.io as sio

from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)

# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


batch_size = 512
original_dim = 512
latent_dim = 12
epochs = 50
subNum = 32
zscore = True

results = []
for latent_dim in range(2, 16):
    
    # encoder
    # inputs = Input(shape=(original_dim, ), name='encoder_input')
    # h = Dense(2048, activation=LeakyReLU(alpha=0.3))(inputs)
    # z_mean = Dense(latent_dim, name='z_mean')(h)
    # z_log_var = Dense(latent_dim, name='z_log_var')(h)
    # z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    # encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    # encoder.summary()

    inputs = Input(shape=(original_dim, ), name='encoder_input')
    x = Dense(512, activation=LeakyReLU(alpha=0.3))(inputs)
    # x = Dense(256, activation=LeakyReLU(alpha=0.3))(x)
    # x = Dense(128, activation=LeakyReLU(alpha=0.3))(x)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    encoder.summary()

    # decoder
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(512, activation=LeakyReLU(alpha=0.3))(latent_inputs)
    # x = Dense(256, activation=LeakyReLU(alpha=0.3))(x)
    # x = Dense(512, activation=LeakyReLU(alpha=0.3))(x)
    x = Dense(original_dim)(x)
    decoder = Model(latent_inputs, x, name='decoder')
    decoder.summary()

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae_mlp')

    # VAE loss = mse_loss or xent_loss + kl_loss
    reconstruction_loss = mean_squared_error(inputs, outputs)
    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)

    rmsprop = optimizers.RMSprop(learning_rate=0.001, rho=0.9, epsilon=None, decay=0.0)
    vae.compile(optimizer=rmsprop)
    vae.summary()
    # plot_model(vae, to_file='vae_mlp.png', show_shapes=True)

    with h5py.File("data/eeg-filtered-normalized.h5", 'r') as f:
        x_train = f.get("data")[:]
    x_test = x_train


    print(type(x_test))

    # train the autoencoder
    model_ = vae.fit(x_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, None))
    
    results.append({'dim':latent_dim, 'loss':model_.history["loss"][-1], 'val_loss':model_.history["val_loss"][-1]})
    
    fig, ax = plt.subplots(5,2)

    for i in range(5):
        input_data = x_train[np.random.randint(len(x_train))].reshape(1, -1)
        encoded = encoder.predict(input_data)
        z = encoded[2]
        decoded = decoder.predict(z)    
        ax[i, 0].plot(input_data[0])
        ax[i, 1].plot(decoded[0])
    

    plt.show()

pp.pprint(results)


    # save the model
    # vae.save_weights('vae_mlp_mnist.h5')

    # build a model to project inputs
    # encoded_x_zmean = encoder.predict(x_train)[0]
    # encoded_x_z = encoder.predict(x_train)[2]
 
    # sio.savemat('D:\\VAE Experiment\\DEAP\\encoded_eegs_2vae\\encoded_eegs_2vae_zmean_sub' +
    #             str(subNo) + '_latentdim' + str(latent_dim) + '.mat',
    #             {'encoded_eegs_zmean': encoded_x_zmean})
    # sio.savemat('D:\\VAE Experiment\\DEAP\\encoded_eegs_2vae\\encoded_eegs_2vae_z_sub' +
    #             str(subNo) + '_latentdim' + str(latent_dim) + '.mat',
    #             {'encoded_eegs_z': encoded_x_z})

    # decoded_x = vae.predict(x_train)
    # sio.savemat('D:\\VAE Experiment\\DEAP\\decoded_eegs_2vae\\decoded_eegs_2vae_sub' +
    #             str(subNo) + '_latentdim' + str(latent_dim) + '.mat',
    #             {'decoded_eegs': decoded_x})