# The aim of this algorithm is to denoise a corrupted set of images taken from the MNIST dataset.
# Autoencoders doesn't have many concrete applications, and this type of algorithm is one of them.
# The process is "simple": original data -> noisy data -> autoencoder -> denoised data

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from keras.layers import Activation, Dense, Input
from keras.layers import Conv2D, Flatten, MaxPooling2D, UpSampling2D
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

np.random.seed(1337)

# data loading (from MNIST dataset)
(xTrain, _), (xTest, _) = mnist.load_data()

imageSize = xTrain.shape[1]
xTrain = np.reshape(xTrain, [-1, imageSize, imageSize, 1])
xTest = np.reshape(xTest, [-1, imageSize, imageSize, 1])
xTrain = xTrain.astype('float32') / 255
xTest = xTest.astype('float32') / 255

# Noise the data
noise = np.random.normal(loc=0.5, scale=0.5, size=xTrain.shape)
xTrainNoisy = xTrain + noise
noise = np.random.normal(loc=0.5, scale=0.5, size=xTest.shape)
xTestNoisy = xTest + noise

xTrainNoisy = np.clip(xTrainNoisy, 0., 1.)
xTestNoisy = np.clip(xTestNoisy, 0., 1.)

# Network parameters
inputShape = (imageSize, imageSize, 1)
batchSize = 128
kernelSize = 3
latentDim = 16
# Encoder/Decoder number of layers and filters per layer
layerFilters = [32, 64]

# build Encoder Model
inputs = Input(shape=inputShape, name='encoder_input')
x = inputs
for filters in layerFilters:
    x = Conv2D(filters=filters,
               kernel_size=kernelSize,
               strides=2,
               activation='relu',
               padding='same')(x)

# Shape info
shape = K.int_shape(x)

# latent vector
x = Flatten()(x)
latent = Dense(latentDim, name='latent_vector')(x)

# Encoder
encoder = Model(inputs, latent, name='encoder')
encoder.summary()

# build Decoder Model
latentInputs = Input(shape=(latentDim,), name='decoder_input')
x = Dense(shape[1] * shape[2] * shape[3])(latentInputs)
x = Reshape((shape[1], shape[2], shape[3]))(x)

for filters in layerFilters[::-1]:
    x = Conv2DTranspose(filters=filters,
                        kernel_size=kernelSize,
                        strides=2,
                        activation='relu',
                        padding='same')(x)

x = Conv2DTranspose(filters=1,
                    kernel_size=kernelSize,
                    padding='same')(x)

outputs = Activation('sigmoid', name='decoder_output')(x)

# Decoder
decoder = Model(latentInputs, outputs, name='decoder')
decoder.summary()

# Autoencoder = Encoder + Decoder
autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
autoencoder.summary()

autoencoder.compile(loss='mse', optimizer='adam')

# Train autoencoder
autoencoder.fit(xTrainNoisy,
                xTrain,
                validation_data=(xTestNoisy, xTest),
                epochs=30,
                batch_size=batchSize)

# Predict the Autoencoder output
xDecoded = autoencoder.predict(xTestNoisy)

# Display the 1st 8 corrupted and denoised images
rows, cols = 10, 30
num = rows * cols
imgs = np.concatenate([xTest[:num], xTestNoisy[:num], xDecoded[:num]])
imgs = imgs.reshape((rows * 3, cols, imageSize, imageSize))
imgs = np.vstack(np.split(imgs, rows, axis=1))
imgs = imgs.reshape((rows * 3, -1, imageSize, imageSize))
imgs = np.vstack([np.hstack(i) for i in imgs])
imgs = (imgs * 255).astype(np.uint8)
plt.figure()
plt.axis('off')
plt.title('Original images: top rows, '
          'Corrupted Input: middle rows, '
          'Denoised Input:  third rows')
plt.imshow(imgs, interpolation='none', cmap='gray')
Image.fromarray(imgs).save('corrupted_and_denoised.png')
plt.show()
