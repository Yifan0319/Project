from __future__ import print_function
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import h5py
import time
import glob
import math
import shutil
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, ZeroPadding2D, Cropping2D, BatchNormalization, Flatten, Bidirectional, Dropout, Reshape,InputSpec
from keras import regularizers

#from keras.engine.topology import Layer, InputSpec
#from keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.base_layer import Layer
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, TensorBoard, CSVLogger
from keras import optimizers
from keras import metrics
from keras.utils import plot_model
import random
from math import *
import json
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from keras import backend as K
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# from sklearn.utils.linear_assignment_ import linear_assignment
from matplotlib.pyplot import savefig
from tensorflow.python.ops import math_ops
from sklearn.cluster import KMeans, AgglomerativeClustering, FeatureAgglomeration, DBSCAN, Birch
from sklearn.mixture import GaussianMixture
import tensorflow as tf
from sklearn.metrics.pairwise import pairwise_distances
import struct
import matplotlib
from matplotlib.ticker import StrMethodFormatter
from matplotlib.ticker import FormatStrFormatter
import datetime
import sys
# from IPython.display import HTML, Image
from matplotlib.animation import FuncAnimation

# Initialize
# train_dataname = r'data\data_processed.npy'
train_dataname = r'/Users/wuyifan/PycharmProjects/CAE model/data_processed.npy'
data_processed = np.load(train_dataname)

# Data process
n, o, p = np.shape(data_processed)
data = np.empty((p, n, o, 1))

for i in range(p):
    data[i, :, :, 0] = data_processed[:, :, i]

# Input shape
img_input = Input(shape=(n, o, 1))  # input shape:[251,16,1]

# Encoder model for CAE
# ---------------------------------------------------------------------------
# intial depth increases by 2x each layer
depth      = 8
strides    = 1
activation = 'relu'
# ReLU initial weights
kernel_initializer='glorot_uniform'
latent_dim = 14

# Crop to dimensions of [120,40,1] to allow for reconstruction
# e = Cropping2D(cropping = ((0, 6), (0, 1)))(img_input)
# e = Conv2D(depth*2**0, (5,5), strides=strides, activation=activation, kernel_initializer=kernel_initializer, padding='same') (e)
e = Conv2D(depth*2**0, (5,5), strides=strides, activation=activation, kernel_initializer=kernel_initializer, padding='same') (img_input)
e = Conv2D(depth*2**1, (5,5), strides=strides, activation=activation, kernel_initializer=kernel_initializer, padding='same') (e)
e = Conv2D(depth*2**2, (3,3), strides=strides, activation=activation, kernel_initializer=kernel_initializer, padding='same') (e)
e = Conv2D(depth*2**3, (3,3), strides=strides, activation=activation, kernel_initializer=kernel_initializer, padding='same')(e)

shape_before_flattening = K.int_shape(e)
x = Flatten()(e)
# Embedded latent space: 14 dimensions (features)
encoded = Dense(latent_dim, activation=activation, name='encoded')(x)

# Decoder model for CAE (Reverse Operations)
# ---------------------------------------------------------------------------
d = Dense(np.prod(shape_before_flattening[1:]), activation='relu')(encoded)
d = Reshape(shape_before_flattening[1:])(d)
d = Conv2DTranspose(depth*2**2, (3,3), strides=strides, activation=activation, kernel_initializer=kernel_initializer, padding='same')(d)
d = Conv2DTranspose(depth*2**1, (5,5), strides=strides, activation=activation, kernel_initializer=kernel_initializer, padding='same')(d)
d = Conv2DTranspose(depth*2**0, (5,5), strides=strides, activation=activation, kernel_initializer=kernel_initializer, padding='same')(d)


decoded = Conv2DTranspose(1, (5,5), strides=strides, activation='linear', kernel_initializer=kernel_initializer, padding='same')(d)

# Define Autoencoder Input: Sprectrograms & Output: Reconstructions
autoencoder = Model(inputs=img_input, outputs=decoded, name='autoencoder')

# Define Encoder Input: Sprectrograms & Output: Compressed image representations (embedded latent space)
encoder = Model(inputs=img_input, outputs=encoded, name='encoder')

# Model architecture
autoencoder.summary()
date_name = '2023'
architecture_fname = 'CAE_Model_{}.png'.format(date_name)
plot_model(autoencoder, to_file=architecture_fname, show_shapes=True)

# Model training parameter
LR = 0.0001  # Learning rate
n_epochs = 600  # Number of epochs
batch_sz = 512  # Batch size

# create log file to record training & validation loss
logger_fname = 'HistoryLog_LearningCurve.csv'.format(date_name)
csv_logger = CSVLogger(logger_fname)

# Early stopping halts training after validation loss stops decreasing for 10 consectutive epochs
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1,
                           mode='min', restore_best_weights=True)

optim = tf.keras.optimizers.Adam(lr=LR)             # Adaptive learning rate optimization algorithm (Adam)
loss = 'mse'                               # Mean Squared Error Loss function

# Compile Encoder & Autoencoder(initialize random filter weights)
encoder.compile(loss=loss, optimizer=optim)
autoencoder.compile(loss=loss,
                  optimizer=optim,
                  metrics=[metrics.mae])

# Model training
tic = time.time()
autoencoder.fit(data, data, batch_size=batch_sz, epochs=n_epochs, callbacks=[csv_logger, early_stop])
toc = time.time()
print('Elapsed Time : {0:4.1f} minutes'.format((toc-tic)/60))

# Model traning result
hist = np.genfromtxt(logger_fname, delimiter=',', skip_header=1, names=['epoch', 'train_mse_loss', 'train_mae_loss'])
plt.figure(figsize=(20,6))

plt.subplot(1,2,2)
plt.plot(hist['epoch'], hist['train_mae_loss'], label='train_mae_loss')
plt.xlabel('Epochs')
plt.title('Training Mean Abs. Error')
plt.legend()

plt.subplot(1,2,1)
plt.plot(hist['epoch'], hist['train_mse_loss'], label='train_mse_loss')
plt.xlabel('Epochs')
plt.title('Training Mean Squared Error')
plt.legend()
plt.show()

