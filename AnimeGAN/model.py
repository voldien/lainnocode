# from ipython_genutils.py3compat import xrange

import tensorflow as tf

# tf.__version__

from tensorflow.keras import layers


def make_generator_model(noise_dim, shape):

	#
	model = tf.keras.Sequential()
	model.add(layers.Dense(8 * 8 * 256, use_bias=False, input_shape=(100, )))
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU())

	model.add(layers.Reshape((8, 8, 256)))
	assert model.output_shape == (None, 8, 8, 256)  # Note: None is the batch size

	model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
	assert model.output_shape == (None, 8, 8, 128)
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU())

	model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
	assert model.output_shape == (None, 16, 16, 64)
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU())

	model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
	assert model.output_shape == (None, 32, 32, 32)
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU())

	model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU())

	model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU())

	assert model.output_shape == (None, shape[0], shape[1], shape[2])

	return model


def make_discriminator_model(shape):
	model = tf.keras.Sequential()
	model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
							input_shape=shape))
	model.add(layers.LeakyReLU())
	model.add(layers.Dropout(0.3))

	model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
	model.add(layers.LeakyReLU())
	model.add(layers.Dropout(0.3))

	model.add(layers.Flatten())
	model.add(layers.Dense(1))

	return model


def discriminator_loss(cross_entropy, real_output, fake_output):
	real_loss = cross_entropy(tf.ones_like(real_output), real_output)
	fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
	total_loss = real_loss + fake_loss
	return total_loss


def generator_loss(cross_entropy, fake_output):
	return cross_entropy(tf.ones_like(fake_output), fake_output)


@tf.function
def train_step(images, **kwargs):

	BATCH_SIZE = kwargs["batch_size"]
	noise_dim = kwargs["noise_dim"]
	generator = kwargs["generator"]
	discriminator = kwargs["discriminator"]
	generator_optimizer = kwargs["generator_optimizer"]
	discriminator_optimizer = kwargs["discriminator_optimizer"]
	cross_entropy = kwargs["cross_entropy"]

	noise = tf.random.normal([BATCH_SIZE, noise_dim])

	with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
		generated_images = generator(noise, training=True)

		real_output = discriminator(images, training=True)
		fake_output = discriminator(generated_images, training=True)

		gen_loss = generator_loss(cross_entropy,fake_output)
		disc_loss = discriminator_loss(cross_entropy, real_output, fake_output)

	gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
	gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

	generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
	discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
