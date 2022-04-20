import GAN



import keras.layers
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LeakyReLU
from keras.layers import Normalization
# tf.__version__
from keras.optimizer_v2.adam import Adam

from tensorflow.keras import layers


def make_generator_model(noise_dim, input_shape, output_shape=(1, 1)):
	model = Sequential()
	# foundation for 4x4 image
	n_nodes = 256 * 4 * 4
	model.add(Dense(n_nodes, input_dim=100))
	# model.add(LeakyReLU(alpha=0.2))
	# model.add(Dense(512))
	# model.add(LeakyReLU(alpha=0.2))
	# model.add(Dense(512))

	# model.add(Dense(512))
	# model.add(LeakyReLU(alpha=0.2))
	# model.add(Dense(n_nodes))

	model.add(Reshape((4, 4, 256)))
	assert model.output_shape == (None, 4, 4, 256)  # Note: None is the batch size

	# upsample to 8x8
	model.add(Conv2DTranspose(512, (4, 4), strides=(1, 1), padding='same'))
	#	model.add(keras.layers.BatchNormalization())
	model.add(keras.layers.ReLU())
	# upsample to 16x16
	model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
	# model.add(keras.layers.BatchNormalization())
	model.add(keras.layers.ReLU())
	# upsample to 32x32
	model.add(Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same'))
	# model.add(keras.layers.BatchNormalization())
	model.add(keras.layers.ReLU())
	# upsample to 64x64
	model.add(Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same'))
	# model.add(keras.layers.BatchNormalization())
	model.add(keras.layers.ReLU())
	# # upsample to 128x128
	# model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
	# model.add(LeakyReLU(alpha=0.2))

	# output layer
	model.add(Conv2DTranspose(3, (3, 3), strides=(2, 2), activation='tanh', padding='same'))
	return model


#
# def make_generator_model(noise_dim, input_shape, output_shape=(1, 1)):
# 	model = Sequential()
# 	# foundation for 4x4 image
# 	n_nodes = 256 * 4 * 4
# 	model.add(Dense(n_nodes, input_dim=100))
# 	# model.add(LeakyReLU(alpha=0.2))
# 	#model.add(Dense(512))
# 	# model.add(LeakyReLU(alpha=0.2))
# 	#model.add(Dense(512))
#
# 	#model.add(Dense(512))
# 	# model.add(LeakyReLU(alpha=0.2))
# 	#model.add(Dense(n_nodes))
#
# 	model.add(Reshape((4, 4, 256)))
# 	assert model.output_shape == (None, 4, 4, 256)  # Note: None is the batch size
#
# 	# upsample to 8x8
# 	model.add(Conv2DTranspose(512, (4, 4), strides=(2, 2), padding='same'))
# 	model.add(keras.layers.BatchNormalization())
# 	model.add(keras.layers.ReLU())
# 	# upsample to 16x16
# 	model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
# 	model.add(keras.layers.BatchNormalization())
# 	model.add(keras.layers.ReLU())
# 	# upsample to 32x32
# 	model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
# 	model.add(keras.layers.BatchNormalization())
# 	model.add(keras.layers.ReLU())
# 	# upsample to 64x64
# 	model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
# 	model.add(keras.layers.BatchNormalization())
# 	model.add(keras.layers.ReLU())
# 	# # upsample to 128x128
# 	# model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
# 	# model.add(LeakyReLU(alpha=0.2))
#
# 	# output layer
# 	model.add(Conv2D(3, (3, 3), activation='tanh', padding='same'))
# 	return model


# def make_discriminator_model(input_shape):
# 	model = Sequential()
# 	# normal
# 	model.add(Conv2D(64, (3, 3), padding='same', input_shape=input_shape))
# 	#model.add(LeakyReLU(alpha=0.2))
# 	# downsample
# 	model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
# 	#model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
# 	#model.add(LeakyReLU(alpha=0.2))
#
# 	# downsample
# 	model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
# 	#model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
#
# 	#model.add(LeakyReLU(alpha=0.2))
# 	# downsample
# 	model.add(Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
# 	#model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
#
# 	# downsample
# 	model.add(Conv2D(512, (3, 3), strides=(2, 2), padding='same'))
# 	#model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
#
# 	model.add(Flatten())
#
# 	model.add(Dense(512))
#
# 	#model.add(LeakyReLU(alpha=0.2))
# 	#model.add(Dropout(0.3))
# 	# classifier
#
# 	#model.add(Dropout(0.4))
# 	model.add(Dense(1))
# 	# compile model
# 	# opt = Adam(lr=0.0002, beta_1=0.5)
# 	# model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
# 	return model


def make_discriminator_model(input_shape):
	model = Sequential()
	# normal
	model.add(Conv2D(64, kernel_size=4, strides=(2, 2), use_bias=False, padding='same', input_shape=input_shape))
	# model.add(keras.layers.BatchNormalization())
	model.add(LeakyReLU(alpha=0.2))
	# downsample
	model.add(Conv2D(128, kernel_size=4, strides=(2, 2), padding='valid', use_bias=False))
	# model.add(keras.layers.BatchNormalization())
	model.add(LeakyReLU(alpha=0.2))

	# downsample
	model.add(Conv2D(256, kernel_size=4, strides=(2, 2), padding='valid', use_bias=False))
	# model.add(keras.layers.BatchNormalization())
	model.add(LeakyReLU(alpha=0.2))
	# downsample
	model.add(Conv2D(512, kernel_size=4, strides=(2, 2), padding='valid', use_bias=False))
	#	model.add(keras.layers.BatchNormalization())
	model.add(LeakyReLU(alpha=0.2))

	model.add(Conv2D(1, kernel_size=4, strides=(2, 2), padding='same', use_bias=False))

	model.add(Flatten())

	model.add(Dense(1, activation='sigmoid'))

	return model