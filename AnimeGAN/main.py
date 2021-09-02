import tensorflow as tf
from pip._vendor.msgpack.fallback import xrange

# tf.__version__

import glob
import imageio
import os.path
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
from numpy import asarray
from IPython import display

import concurrent.futures as cf

from AnimeGAN.dataProcessing import loadDatSet
from AnimeGAN.model import make_generator_model, make_discriminator_model, train_step


def train(dataset, epochs, checkpoint):
	for epoch in range(epochs):
		start = time.time()

		for image_batch in dataset:
			train_step(image_batch, batch_size=BATCH_SIZE, noise_dim=noise_dim, generator=generator,
					   discriminator=discriminator, generator_optimizer=generator_optimizer,
					   discriminator_optimizer=discriminator_optimizer, cross_entropy=cross_entropy
					   )

		# Produce images for the GIF as we go
		# display.clear_output(wait=True)
		generate_and_save_images(generator,
								 epoch + 1,
								 seed)

		# Save the model every 15 epochs
		if (epoch + 1) % 15 == 0:
			checkpoint.save(file_prefix=checkpoint_prefix)

		print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

	# Generate after the final epoch
	# display.clear_output(wait=True)
	generate_and_save_images(generator,
							 epochs,
							 seed)


def generate_and_save_images(model, epoch, test_input):
	# Notice `training` is set to False.
	# This is so all layers run in inference mode (batchnorm).
	predictions = model(test_input, training=False)

	fig = plt.figure(figsize=(4, 4))

	for i in range(predictions.shape[0]):
		plt.subplot(4, 4, i + 1)
		rgb = predictions[i, :, :, 0:3] * 127.5 + 127.5
		rgb /= 255.0
		plt.imshow(asarray(rgb))
		plt.axis('off')

	plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))


# plt.show()


# Press the green button in the gutter to run the script.
def processImageDataset(train_images):
	# train_images = train_images.astype('float32')

	# Do per section of the
	norm1 = []
	for i in range(len(train_images)):
		norm1.append(train_images[i].astype('float32') / np.linalg.norm(train_images[i].astype('float32')))
	return np.array(norm1)
	pass


if __name__ == '__main__':
	dataset_files = ["/media/data-sets/animeface.zip", "/media/data-sets/anime-face-dataset.zip"]

	(train_images, train_labels), (_, _) = loadDatSet(
		dataset_files[0:1])

	train_images = processImageDataset(train_images[0:3000])
	print(len(train_images))

	BUFFER_SIZE = train_images.shape[0]
	BATCH_SIZE = 256

	train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

	noise = tf.random.normal([1, 100])

	generator = make_generator_model([1, 100], train_images[0].shape)
	# generator.model()

	generated_image = generator(noise, training=False)

	plt.imshow(generated_image[0, :, :, 0], cmap='gray')

	discriminator = make_discriminator_model(train_images[0].shape)
	decision = discriminator(generated_image)
	print(decision)

	cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

	#
	generator_optimizer = tf.keras.optimizers.Adam(1e-4)
	discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

	#
	checkpoint_dir = './training_checkpoints'
	checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
	checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
									 discriminator_optimizer=discriminator_optimizer,
									 generator=generator,
									 discriminator=discriminator)
	EPOCHS = 50
	noise_dim = 100
	num_examples_to_generate = 16

	seed = tf.random.normal([num_examples_to_generate, noise_dim])

	train(train_dataset, EPOCHS, checkpoint)

	checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
