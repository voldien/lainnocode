import zipfile
from io import StringIO, BytesIO  ## for Python 3
# from tensorboard.plugins.hparams import api as hp

import zipfile
from io import StringIO, BytesIO  ## for Python 3
from tensorboard.plugins.hparams import api as hp

import PIL.Image

# from ipython_genutils.py3compat import xrange

import tensorflow as tf
from pip._vendor.msgpack.fallback import xrange

# tf.__version__

import glob
from tensorboard.plugins.hparams import api as hp
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

from AnimeGAN.model import make_generator_model, make_discriminator_model


def loadImageDataSubSet(path, subset):
	images = []
	_n = int(len(subset))
	with zipfile.ZipFile(path, 'r') as zip:
		for i in range(_n):
			file_in_zip = subset[i]
			if (".jpg" in file_in_zip or ".JPG" in file_in_zip or ".png" in file_in_zip):
				data = zip.read(file_in_zip)
				stream = BytesIO(data)
				image = PIL.Image.open(stream)
				image = image.resize((128, 128), PIL.Image.BILINEAR)
				images.append(asarray(image))
				stream.close()
	return images


def load_image_data(pool, path, size):
	future_to_image = []
	with zipfile.ZipFile(path, 'r') as zip:
		zlist = zip.namelist()
		nr_chunks = 32
		chunk_size = int(len(zlist) / nr_chunks)
		for i in range(nr_chunks):
			subset = zlist[chunk_size * i: chunk_size * (i + 1)]
			task = pool.submit(loadImageDataSubSet, path, subset)
			future_to_image.append(task)
	return future_to_image


def loadDatSet(paths, filter=None, ProcessOverride=None, size=(128, 128)):
	future_to_image = []
	total_data = []
	with cf.ProcessPoolExecutor() as pool:
		for path in paths:
			for f in load_image_data(pool, path, size):
				future_to_image.append(f)
		for future in cf.as_completed(set(future_to_image)):
			try:
				data = future.result()
				for x in data:
					total_data.append(x)
			except Exception as exc:
				print('%r generated an exception: %s' % ("url", exc))
			else:
				print('%r page is %d bytes' % ("url", len(data)))
			del data
	return (np.array(total_data), None), (None, None)



def train(dataset, epochs):
	for epoch in range(epochs):
		start = time.time()

		for image_batch in dataset:
			train_step(image_batch)

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
	train_images = train_images.astype('float32')

	norm1 = train_images / np.linalg.norm(train_images)
	return norm1
	pass


if __name__ == '__main__':
	(train_images, train_labels), (_, _) = loadDatSet(
		["/media/data-sets/animeface.zip", "/media/data-sets/anime-face-dataset.zip"])

	train_images = processImageDataset(train_images)
	print(len(train_images))

	BUFFER_SIZE = train_images.shape[0]
	BATCH_SIZE = 256

	train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

	generator = make_generator_model()
	generator.model()

	noise = tf.random.normal([1, 100])
	with tf.summary.create_file_writer(run_dir).as_default():
		hp.hparams(hparams)
	generated_image = generator(noise, training=False, )

	plt.imshow(generated_image[0, :, :, 0], cmap='gray')

	discriminator = make_discriminator_model()
	decision = discriminator(generated_image)
	print(decision)

	cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

	generator_optimizer = tf.keras.optimizers.Adam(1e-4)
	discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

	checkpoint_dir = '../training_checkpoints'
	checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
	checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
									 discriminator_optimizer=discriminator_optimizer,
									 generator=generator,
									 discriminator=discriminator)
	EPOCHS = 50
	noise_dim = 100
	num_examples_to_generate = 16

	seed = tf.random.normal([num_examples_to_generate, noise_dim])
