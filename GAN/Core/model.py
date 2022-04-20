# from ipython_genutils.py3compat import xrange
import math

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
from keras.layers import Activation
# tf.__version__
from keras.optimizer_v2.adam import Adam

from tensorflow.keras import layers


# TODO add option to set resolution.

def make_generator_model(input_shape, output_shape=(1, 1), **kwargs):
	model = Sequential()
	init = tf.keras.initializers.RandomNormal(stddev=0.02)
	num_layers = max(int(math.log2(output_shape[0])) - 4, 0)

	# foundation for 4x4 image
	# TODO adjust nodes based on resolution.
	# TODO add input_shape
	n_nodes = 1024 * 4 * 4
	model.add(Dense(n_nodes, kernel_initializer=init, input_dim=input_shape))
	model.add(layers.BatchNormalization())
	model.add(keras.layers.ReLU())

	model.add(Dense(n_nodes, kernel_initializer=init))
	model.add(layers.BatchNormalization())
	model.add(keras.layers.ReLU())

	model.add(Reshape((4, 4, 1024)))
	assert model.output_shape == (None, 4, 4, 1024)  # Note: None is the batch size

	# upsample to 8x8
	model.add(Conv2DTranspose(512, (4, 4), strides=(2, 2), use_bias=False, padding='same', kernel_initializer=init))
	model.add(keras.layers.BatchNormalization())
	model.add(keras.layers.ReLU())

	# TODO verify it is a power of two

	for i in range(0, num_layers):
		filter_size = int(512 / (2 ** i))
		image_size = (16 * (2 ** i), 16 * (2 ** i))
		model.add(Conv2DTranspose(filter_size, (4, 4), strides=(2, 2), use_bias=False, padding='same',
								  kernel_initializer=init))
		model.add(keras.layers.BatchNormalization())
		model.add(keras.layers.ReLU())
		assert model.output_shape == (None, image_size[0], image_size[1], filter_size)

	# output layer
	model.add(Conv2DTranspose(3, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init))
	model.add(layers.Activation('tanh'))
	# TODO add color channel
	assert model.output_shape == (None, output_shape[0], output_shape[1], 3)
	return model


def make_discriminator_model(input_shape):
	model = Sequential()

	n_layers = max(int(math.log2(input_shape[1])) - 3, 0)

	model.add(Conv2D(4, kernel_size=64, strides=(2, 2), use_bias=False, padding='same',
					 input_shape=input_shape))
	model.add(keras.layers.BatchNormalization())
	model.add(LeakyReLU(alpha=0.2))

	for i in range(0, n_layers):
		filter_size = 128 * (i + 1)
		kernel_size = 4
		model.add(Conv2D(filter_size, kernel_size=kernel_size, strides=(2, 2), use_bias=False, padding='same'))
		model.add(keras.layers.BatchNormalization())
		model.add(LeakyReLU(alpha=0.2))

	model.add(Conv2D(1, kernel_size=4, strides=(2, 2), padding='valid', use_bias=False))
	model.add(Flatten())
	model.add(Activation('sigmoid'))

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


#
# # normal
# model.add(Conv2D(64, kernel_size=4, strides=(2, 2), use_bias=False, padding='same', input_shape=input_shape))
# model.add(keras.layers.BatchNormalization())
# model.add(LeakyReLU(alpha=0.2))
#
# # downsample
# model.add(Conv2D(128, kernel_size=4, strides=(2, 2), padding='same', use_bias=False))
# model.add(keras.layers.BatchNormalization())
# model.add(LeakyReLU(alpha=0.2))
#
# # downsample
# model.add(Conv2D(256, kernel_size=4, strides=(2, 2), padding='same', use_bias=False))
# model.add(keras.layers.BatchNormalization())
# model.add(LeakyReLU(alpha=0.2))
#
# # downsample
# model.add(Conv2D(512, kernel_size=5, strides=(2, 2), padding='same', use_bias=False))
# model.add(keras.layers.BatchNormalization())
# model.add(LeakyReLU(alpha=0.2))

# # downsample
# model.add(Conv2D(1024, kernel_size=6, strides=(2, 2), padding='same', use_bias=False))
# model.add(keras.layers.BatchNormalization())
# model.add(LeakyReLU(alpha=0.2))
#
# # downsample
# model.add(Conv2D(512, kernel_size=4, strides=(2, 2), padding='valid', use_bias=False))
# #	model.add(keras.layers.BatchNormalization())
# model.add(LeakyReLU(alpha=0.2))


#
#
# def discriminator_loss(cross_entropy, real_output, fake_output):
# 	real_loss = cross_entropy(tf.ones_like(real_output), real_output)
# 	fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
# 	total_loss = real_loss + fake_loss
# 	return total_loss
#
#
# def generator_loss(cross_entropy, fake_output):
# 	return cross_entropy(tf.ones_like(fake_output), fake_output)
#
#
# @tf.function
# def train_step(dataset, generator, discriminator, generator_optimizer, discriminator_optimizer, cross_entropy):
# 	noise = tf.random.normal([32, 100])
#
# 	with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
# 		generated_images = generator(noise, training=True)
#
# 		real_output = discriminator(dataset, training=True)
# 		fake_output = discriminator(generated_images, training=True)
#
# 		gen_loss = generator_loss(cross_entropy, fake_output)
# 		disc_loss = discriminator_loss(cross_entropy, real_output, fake_output)
#
# 	gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
# 	gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
#
# 	generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
# 	discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
#
# 	return gen_loss, disc_loss
