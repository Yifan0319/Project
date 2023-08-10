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
from keras.engine.base_layer import Layer
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, TensorBoard, CSVLogger
from keras import optimizers
#from keras.optimizers.legacy import *
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
from sklearn.utils.linear_assignment_ import linear_assignment
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
from IPython.display import HTML, Image
from matplotlib.animation import FuncAnimation

#Initialize
train_dataname = r'data\data_processed.npy'
data_processed = np.load(train_dataname)

#Data process
p,n,o ,channel = np.shape(data_processed)
data = np.empty((p,n,o,channel))

# for i in range(p):
#     data[i,:,:,0] = data_processed[i,:,:,:]
data = data_processed

# Scale Data -  rm mean and scale each by stanrdard deviation of each feature (pixel)
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True)
# fit the scaler to the training data
datagen.fit(data)
# scale the data
data = datagen.standardize(data)

# Split data into training data(80%) and validation data (20%)
X_train, X_val = train_test_split(data, test_size=0.2,
                                  shuffle=True,
                                  random_state=812)
#Ipnut shape
img_input = Input(shape=(n, o, 3)) # input shape:[126,41,1]

#Encoder model for CAE
#---------------------------------------------------------------------------
# intial depth increases by 2x each layer
depth      = 8
strides    = 1
activation = 'relu'
# ReLU initial weights
kernel_initializer='glorot_uniform'
latent_dim = 14

#Crop to dimensions of [120,40,1] to allow for reconstruction
#e = Cropping2D(cropping = ((0, 6), (0, 1)))(img_input)
#e = Conv2D(depth*2**0, (5,5), strides=strides, activation=activation, kernel_initializer=kernel_initializer, padding='same') (e)
e = Conv2D(depth*2**0, (5,5), strides=strides, activation=activation, kernel_initializer=kernel_initializer, padding='same') (img_input)
e = Conv2D(depth*2**1, (5,5), strides=strides, activation=activation, kernel_initializer=kernel_initializer, padding='same') (e)
e = Conv2D(depth*2**2, (3,3), strides=strides, activation=activation, kernel_initializer=kernel_initializer, padding='same') (e)
e = Conv2D(depth*2**3, (3,3), strides=strides, activation=activation, kernel_initializer=kernel_initializer, padding='same')(e)

shape_before_flattening = K.int_shape(e)
x = Flatten()(e)
#Embedded latent space: 14 dimensions (features)
encoded = Dense(latent_dim, activation=activation, name='encoded')(x)

#Decoder model for CAE (Reverse Operations)
#---------------------------------------------------------------------------
d = Dense(np.prod(shape_before_flattening[1:]), activation='relu')(encoded)
d = Reshape(shape_before_flattening[1:])(d)
d = Conv2DTranspose(depth*2**2, (3,3), strides=strides, activation=activation, kernel_initializer=kernel_initializer, padding='same')(d)
d = Conv2DTranspose(depth*2**1, (5,5), strides=strides, activation=activation, kernel_initializer=kernel_initializer, padding='same')(d)
d = Conv2DTranspose(depth*2**0, (5,5), strides=strides, activation=activation, kernel_initializer=kernel_initializer, padding='same')(d)


decoded = Conv2DTranspose(3, (5,5), strides=strides, activation='linear', kernel_initializer=kernel_initializer, padding='same')(d)

# Define Autoencoder Input: Sprectrograms & Output: Reconstructions
autoencoder = Model(inputs=img_input, outputs=decoded, name='autoencoder')

#Define Encoder Input: Sprectrograms & Output: Compressed image representations (embedded latent space)
encoder = Model(inputs=img_input, outputs=encoded, name='encoder')

#Model architecture
autoencoder.summary()
date_name = '2023'
architecture_fname = 'CAE_Model_{}.png'.format(date_name)
plot_model(autoencoder, to_file=architecture_fname, show_shapes=True)

#Model training parameter
LR = 0.0001  # Learning rate
n_epochs = 40  # Number of epochs
batch_sz = 50  # Batch size

# create log file to record training & validation loss
logger_fname = 'HistoryLog_LearningCurve.csv'.format(date_name)
csv_logger = CSVLogger(logger_fname)

# Early stopping halts training after validation loss stops decreasing for 10 consectutive epochs
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1,
                           mode='min', restore_best_weights=True)

optim = tf.keras.optimizers.Adam(lr=LR)             #Adaptive learning rate optimization algorithm (Adam)
loss = 'mse'                               #Mean Squared Error Loss function

#Compile Encoder & Autoencoder(initialize random filter weights)
encoder.compile(loss=loss,optimizer=optim)
autoencoder.compile(loss=loss,
                  optimizer=optim,
                  metrics=[metrics.mae])

#Model training
tic=time.time()
autoencoder.fit(X_train, X_train, batch_size=batch_sz, epochs=n_epochs, validation_data=(X_val, X_val), callbacks=[csv_logger, early_stop])
toc = time.time()
print('Elapsed Time : {0:4.1f} minutes'.format((toc-tic)/60))

#Model traning result
hist = np.genfromtxt(logger_fname, delimiter=',', skip_header=1, names=['epoch', 'train_mse_loss', 'train_mae_loss', 'val_mse_loss', 'val_mae_loss'])
plt.figure(figsize=(20,6))

plt.subplot(1,2,2)
plt.plot(hist['epoch'], hist['train_mae_loss'], label='train_mae_loss')
plt.plot(hist['epoch'], hist['val_mae_loss'], label='val_mae_loss')
plt.xlabel('Epochs')
plt.title('Training Mean Abs. Error')
plt.legend()

plt.subplot(1,2,1)
plt.plot(hist['epoch'], hist['train_mse_loss'], label='train_mse_loss')
plt.plot(hist['epoch'], hist['val_mse_loss'], label='val_mse_loss')
plt.xlabel('Epochs')
plt.title('Training Mean Squared Error')
plt.legend()

plt.show()

val_reconst = autoencoder.predict(X_val, verbose = 1) #reconstruction of validation data
val_enc = encoder.predict(X_val, verbose = 1)         #embedded latent space samples of validation data
enc_train = encoder.predict(X_train, verbose = 1)     #embedded latent space samples of training data

#Axis labels to represent time in seconds rather than time bins from S.T.F.T.
secs = ['0', '1', '2', '3', '4']
secs_pos = np.arange(0,50,10)

good_idx = [17343, 225302, 170169, 394137]  # samples to compare (Useful in previous models, arbitrary now)
cnt = 0
fig = plt.figure(figsize=(13, 10))
for imgIdx in np.random.randint(0, len(X_val), 4):
    cnt = cnt + 1
    # Top row shows original input spectrograms
    fig.add_subplot(2, 4, cnt)
    plt.imshow(X_val[imgIdx, :, :, 0])
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.ylabel('Frequency (Hz)')
    plt.xticks(secs_pos, secs)
    plt.xlabel('Time(s)')
    plt.title(f'Original idx {imgIdx}')

    # Bottom row shows the reconstructed spectrograms (CAE output)
    fig.add_subplot(2, 4, cnt + 4)
    plt.imshow(val_reconst[imgIdx,:,:,0])
    plt.colorbar()
    plt.title(f'Reconstructed idx {imgIdx}')
    plt.gca().invert_yaxis()
    plt.ylabel('Frequency (Hz)')
    plt.xticks(secs_pos, secs)
    plt.xlabel('Time(s)')
plt.show()
plt.close()

from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(12, 8))
idx = 21

gs = GridSpec(nrows=3, ncols=3, width_ratios=[1, .5, 1], height_ratios=[1, 0, 0])
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])
ax2 = plt.subplot(gs[2])

# Original spectrogram
cb0 = ax0.imshow(X_val[idx, :, :, 1])
ax0.set_ylabel('Frequency (Hz)')
ax0.set_xticks(secs_pos, secs)
ax0.set_xlabel('Time(s)')
ax0.invert_yaxis()
plt.colorbar(cb0, ax=ax0)
ax0.set_title('Original Spectrogram')

# Latent space image representation
cb1 = ax1.imshow(val_enc[idx].reshape(14, 1), cmap='viridis')
ax1.invert_yaxis()
ax1.set_aspect(2)
plt.colorbar(cb1, ax=ax1)
ax1.set_title('Latent Space')

# Reconstructed Image
cb2 = ax2.imshow(val_reconst[idx, :, :, 1])
ax2.set_ylabel('Frequency (Hz)')
ax2.set_xticks(secs_pos, secs)
ax2.set_xlabel('Time(s)')
ax2.invert_yaxis()
plt.colorbar(cb2, ax=ax2)
ax2.set_title('Reconstructed Spectrogram')

fig.tight_layout()
plt.show()


#For sake of time and computational efficiency-take 50,000 random samples
#from training data to compute gap statistic
rand_idx = np.random.randint(0,len(X_train),500)
kmeans_enc = np.zeros([500,14])
for i in range(len(rand_idx)):
    kmeans_enc[i] = enc_train[rand_idx[i]]
print(kmeans_enc.shape)

# Calculate min, max, mean, and standard deviation of each features in embedded latent space samples
feat_min = np.amin(kmeans_enc, axis=0)
feat_max = np.amax(kmeans_enc, axis=0)
feat_mean = np.mean(kmeans_enc, axis=0)
feat_std = np.std(kmeans_enc, axis=0)

# Generate a Gaussian normal distribtuion with 50,000 samples based on feature mean and standard deviaton .
gauss = np.zeros([len(feat_mean), len(kmeans_enc)])

# Generate a uniform reference distribtuion with 50,000 samples based on feature min and max.
uniform = np.zeros([len(feat_mean), len(kmeans_enc)])

for i in range(0, len(feat_mean)):
    gauss[i] = np.random.normal(loc=feat_mean[i], scale =feat_std[i], size=len(kmeans_enc))
    uniform[i] = np.random.uniform(low=feat_min[i], high=feat_max[i], size=len(kmeans_enc))

# Run Kmeans with a range of number of clusters between (2-20) on embedded latent space training samples
# for each number of clusters, Kmeans finds the cluster centers that minimizes the sum-of-squares distance (intertia)
# between the samples assigned to a cluster, and that cluster centroid and the cluster center

tic = time.time()
inertia = []
for i in np.arange(2, 20, 1):
    kmeans_model_test = KMeans(n_clusters=i, n_init=10, precompute_distances=True, random_state=812, verbose=0).fit(
        kmeans_enc)
    inertia.append(kmeans_model_test.inertia_)

np.savez('Kmeans_inertia_{}.npz'.format(date_name), inertia)
toc = time.time()
print('Latent Space Representation KMeans Computation Time : {0:4.1f} minutes'.format((toc - tic) / 60))

def print_cluster_size(labels):
    """
    Shows the number of samples assigned to each cluster.
    # Example
    ```
        print_cluster_size(labels=kmeans_labels)
    ```
    # Arguments
        labels: 1D array  of clustering assignments. The value of each label corresponds to the cluster
                that the samples in the clustered data set (with the same index) was assigned to. Array must be the same length as
                data.shape[0]. where 'data' is the clustered data set.
    """
    num_labels = max(labels) + 1
    for j in range(0,num_labels):
        label_idx = np.where(labels==j)[0]
        print("Label " + str(j) + ": " + str(label_idx.shape[0]))


def print_all_clusters(data, labels, num_clusters):
    """
    Shows six examples of spectrograms assigned to each cluster.
    # Example
    ```
        print_all_clusters(data=X_train, labels=kmeans_labels, num_clusters=10)
    ```
    # Arguments
        data: data set (4th rank tensor) that was used as the input for the clustering algorithm used.
        labels: 1D array  of clustering assignments. The value of each label corresponds to the cluster
                that the samples in 'data' (with the same index) was assigned to. Array must be the same length as
                data.shape[0].
        num_clusters: The number of clusters that the data was seperated in to.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """
    fig1 = plt.figure()
    for cluster_num in range(0, num_clusters):
        fig = plt.figure(figsize=(14, 5))
        num_labels = max(labels) + 1
        cnt = 0
        label_idx = np.where(labels == cluster_num)[0]

        if len(label_idx) < 6:
            for i in range(len(label_idx)):
                fig.add_subplot(1, 6, i + 1)
                plt.imshow(data[label_idx[i], : , : ,1])
                plt.ylabel('Frequency (Hz)')
                plt.xticks(secs_pos, secs)
                plt.xlabel('Time(s)')
                plt.gca().invert_yaxis()
                plt.colorbar()
        else:
            for i in range(0, 6):
                cnt = cnt + 1
                fig.add_subplot(1, 6, cnt)
                plt.imshow(data[label_idx[i], : , : ,1])
                plt.ylabel('Frequency (Hz)')
                plt.xticks(secs_pos, secs)
                plt.xlabel('Time(s)')
                plt.gca().invert_yaxis()
                plt.colorbar()

        plt.suptitle('Label {}'.format(cluster_num), ha='left', va='center', fontsize=28)
        plt.tight_layout()

    plt.show()

n_clusters = 5 #From Uniform Gap Test (very subtle elbow)

optimal = tf.optimizers.legacy.Adam(lr=LR)
#### clustering layers:
class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.
    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):  # self is place holder for futre object
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        # initialize object attributes
        self.n_clusters = n_clusters
        self.alpha = alpha  # exponent for soft assignment calculation
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


print('...Finetuning...')
clustering_layer = ClusteringLayer(n_clusters, name='clustering')(encoder.output)  # Feed embedded samples to
# clustering layer

model = Model(inputs=autoencoder.input, outputs=[clustering_layer, autoencoder.output])  # Input: Spectrograms,
# Output: Cluster assignments
#      & Reconstructions

model.compile(loss=['kld', loss], loss_weights=[0.1, .9], optimizer=optimal)  # Initialize model parameters


enc_train = encoder.predict(X_train, verbose=1)  # generate embedded latent space training samples

### initializing the weights using Kmean and assigning them to the model
# ---------------------------------------------------------------------------------------------------------------------------
kmeans = KMeans(n_clusters=n_clusters,
                n_init=100)  # run kmeans with n_clusters, run 100 initializations to ensure accuracy

labels = kmeans.fit_predict(enc_train)  # get initial assignments

labels_last = np.copy(labels)  # make copy of labels for future reference (see DEC training below)

# initialize the DEC clustering layer weights using cluster centers found initally by kmeans.
model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

# Parameters for the  DEC finetuning
#--------------------------------------------------------------------------------------------------------------------
batch_size= 30                     # number of samples in each batch
tol = 0.001                        # tolerance threshold to stop training
loss = 0                           # initialize loss
index = 0                          # initialize index to start
maxiter = 50                    # number of updates to rub before halting. (~12 epochs)
update_interval = 5              # Soft assignment distribution and target distributions updated evey 315 batches.
                                   #(~12 updates/epoch)
index_array = np.arange(X_train.shape[0])


###############################################################################
### simultaneous optimization and clustering
def target_distribution(q):
    """
    Compute the target distribution p, given soft assignements, q. The target distribtuion is generated by giving
    more weight to 'high confidence' samples - those with a higher probability of being a signed to a certain cluster.
    This is used in the KL-divergence loss function.
    # Arguments
        q: Soft assignement probabilities - Probabilities of each sample being assigned to each cluster.
    # Input:
         2D tensor of shape [n_samples, n_features].
    # Output:
        2D tensor of shape [n_samples, n_features].
    """
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T


tic = time.time()
loss_list = np.zeros([maxiter, 3])  # Keep track of loss function during training process
for ite in range(int(maxiter)):
    if ite % update_interval == 0:
        q, reconst = model.predict(X_train, verbose=1)  # Calculate soft assignment distribtuion & CAE reconstructions

        p = target_distribution(q)  # Update the auxiliary target distribution p

        labels = q.argmax(1)  # Assign labels to the embedded latent space samples

        # check stop criterion - Calculate the % of labels that changed from previous update
        delta_label = np.sum(labels != labels_last).astype(np.float32) / labels.shape[0]

        labels_last = np.copy(labels)  # Generate copy of labels for future updates

        loss = np.round(loss, 5)  # Round the loss

        print('Iter %d' % ite)
        print('Loss: {}'.format(loss))
        print_cluster_size(labels)  # Show the number of samples assigned to each cluster

        if ite > 0 and delta_label < tol:  # Break training if loss reaches the tolerance threshhold
            print('delta_label ', delta_label, '< tol ', tol)
            break

    idx = index_array[index * batch_size: min((index + 1) * batch_size, X_train.shape[0])]
    loss = model.train_on_batch(x=X_train[idx], y=[p[idx], X_train[idx, :, :, :]])
    index = index + 1 if (index + 1) * batch_size <= X_train.shape[0] else 0

# Save model and model weights seperately
model.save_weights('./DEC_model_final_{}.h5'.format(date_name))
model.save('Saved_DEC_model_{}.hdf5'.format(date_name))

toc = time.time()
print('Deep Embedded Clustering Computation Time : {0:4.1f} minutes'.format((toc - tic) / 60))

cnt = 0
fig = plt.figure(figsize=(13, 10))
for imgIdx in np.random.randint(0, len(X_train), 4):
    cnt = cnt + 1
    fig.add_subplot(2, 4, cnt)
    plt.imshow(X_train[imgIdx, :, :, 1])
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.ylabel('Frequency (Hz)')
    plt.xticks(secs_pos, secs)
    plt.xlabel('Time(s)')
    plt.title(f'Original idx {imgIdx}')

    fig.add_subplot(2, 4, cnt + 4)
    plt.imshow(reconst[imgIdx, :, :, 1])
    plt.colorbar()
    plt.title(f'Reconstructed idx {imgIdx}')
    plt.gca().invert_yaxis()
    plt.ylabel('Frequency (Hz)')
    plt.xticks(secs_pos, secs)
    plt.xlabel('Time(s)')

plt.show()
plt.close()

X_test = X_val##revise
q, reconst = model.predict(X_test, verbose = 1)#Predict asignment probability of test data & generate reconstructions
labels = q.argmax(1)                           #Determine labels based on assignment probabilities
enc_test = encoder.predict(X_test)             #Generate embedded latent space test data samples

# Show examples of original spectrograms and reconstructions.
cnt = 0
fig = plt.figure(figsize=(13, 10))
for imgIdx in np.random.randint(0, len(X_test), 4):
    cnt = cnt + 1
    fig.add_subplot(2, 4, cnt)
    plt.imshow(X_test[imgIdx, :, :, 1])
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.ylabel('Frequency (Hz)')
    plt.xticks(secs_pos, secs)
    plt.xlabel('Time(s)')
    plt.title(f'Original idx {imgIdx}')

    fig.add_subplot(2, 4, cnt + 4)
    plt.imshow(reconst[imgIdx,:,:,1])  # , vmin = c_min, vmax = c_max/2)
    plt.colorbar()
    plt.title(f'Reconstructed idx {imgIdx}')
    plt.gca().invert_yaxis()
    plt.ylabel('Frequency (Hz)')
    plt.xticks(secs_pos, secs)
    plt.xlabel('Time(s)')
plt.show()
plt.close()

# Show the number of samples assigned to each cluster
print_cluster_size(labels)

# Show six examples of sample spectrogram assigned to each cluster
print_all_clusters(X_test, labels, n_clusters)