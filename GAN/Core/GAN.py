#!/usr/bin/env python3
import argparse
import logging
import os
import pathlib
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from numpy import asarray

from DCGANModel import DCGAN
from util.image import generate_image
from model import make_generator_model, make_discriminator_model
from util.dataProcessing import loadImagedataSet


def load_from_directory(data_dir, args):
	def configure_for_performance(ds, AUTOTUNE):
		ds = ds.cache()
		ds = ds.prefetch(buffer_size=AUTOTUNE)
		return ds

	image_count = len(list(data_dir.glob('**/*.??g')))
	print("{0}: Found {1}".format(data_dir, image_count))

	#
	train_ds = tf.keras.utils.image_dataset_from_directory(
		data_dir,
		interpolation='bilinear',
		color_mode='rgb',
		label_mode=None,
		shuffle=True,
		image_size=args.image_size,
		batch_size=args.batch_size)
	#
	AUTOTUNE = tf.data.AUTOTUNE
	normalization_layer = tf.keras.layers.Rescaling(1. / 255.0)

	# Translate [0,1] -> [-1, 1]
	normalized_ds = configure_for_performance(train_ds.map(lambda x: normalization_layer(x) * 2.0 - 1.0),
											  AUTOTUNE)

	return normalized_ds


def load_zip_file(path, args):
	pass


def load_all_datasets(args):
	# dataset_files = args.data_sets_file_paths
	for path in dataset_files:
		cache_path = "{0}.npz".format(pathlib.Path(path).name)
		data_dir = pathlib.Path(path)
		if os.path.isdir(path):
			return load_from_directory(data_dir)
		elif data_dir.suffix == 'zip':
			if not os.path.exists(cache_path):
				print("loading dataset archive {0}".format(path))
				(train_images, train_labels), (_, _) = loadImagedataSet(path, size=args.image_size)

				#
				if train_images.shape[-1] > 3:
					train_images = np.expand_dims(train_images, axis=-1)  # <--- add batch axis

				train_images = train_images.astype('float32')
				# Translate [0,1] -> [-1, 1]
				train_images = 2.0 * (train_images * (1.0 / 255.0)) - 1.0

				with open(cache_path, 'wb') as f:
					np.savez_compressed(f, train=train_images)
				del train_images


def generate_and_save_images(model, epoch, seed):
	# Notice `training` is set to False.
	# This is so all layers run in inference mode (batchnorm).
	# Generaste the image.
	predictions = generate_image(model, seed)

	# Generate the figure frame.
	fig = plt.figure(figsize=(4, 4))
	#
	for i in range(predictions.shape[0]):
		plt.subplot(4, 4, i + 1)
		rgb = (predictions[i, :, :, 0:3] + 1.0) / 2.0
		# rgb /= 255.0
		plt.imshow(asarray(rgb))
		plt.axis('off')

	fig.savefig('image_at_epoch_{:04d}.png'.format(epoch))


def show_images(images, nmax=64):
	fig, ax = plt.subplots(figsize=(8, 8))
	ax.set_xticks([])
	ax.set_yticks([])


# ax.imshow(make_grid(denorm(images.detach()[:nmax]), nrow=8).permute(1, 2, 0))


def show_batch(dl, nmax=64):
	for images, _ in dl:
		show_images(images, nmax)
		break


def plotCostHistory(history, loss_label="", val_label="", title="", x_label="", y_label=""):
	for k, v in history.items():
		plt.plot(v, label=k)
	plt.title(label=title)
	plt.ylabel(ylabel=y_label)
	plt.xlabel(xlabel=x_label)
	plt.legend(loc="upper left")
	plt.show(block=False)


class checkpoint_callback(tf.keras.callbacks.Callback):
	"""
    Subclass of keras.callbacks.Callback to save the weights every epoch in a .h5 file
    """

	def __init__(self, **kwargs):
		super(tf.keras.callbacks.Callback, self).__init__(**kwargs)

	def on_epoch_end(self, epoch, logs=None):
		#TODO fix!
		pass
		#self.model.generator.save_weights("Weights/generator_weights-test.h5")
		#self.model.discriminator.save_weights("Weights/discriminator_weights-test.h5")


class save_images(tf.keras.callbacks.Callback):
	"""
	This is a subclass of the keras.callbacks.Callback class.
	On subclassing it we can specify methods which can be executed while training
	"""

	def __init__(self, noise, **kwargs):
		super(tf.keras.callbacks.Callback, self).__init__(**kwargs)
		self.noise = noise


	# overwriting on_epoch_end() helps in executing a custom method when an epoch ends
	def on_epoch_end(self, epoch, logs=None):
		"""
		Saves images generated from a fixed random vector by the generator to the disk

		Parameters:
			noise: fixed noise vector from a normal distribution to be fed to the generator.
			num_rows: number of rows of images
			num_cols: number of columns of images
			margin: margin between images
			generator: keras model representing the generator network

		"""
		generate_and_save_images(self.model.generator, epoch, self.noise)


def run_train_model(args, train_images):
	# TODO make it only accept batched train data object.

	# if create model
	if args.data_sets_file_paths:
		pass

	logging.info(len(train_images))

	# Extract the image size.

	tensor_train_dataset = train_images

	image_batch = next(iter(tensor_train_dataset))
	image_size = image_batch[0].shape
	assert image_size == (args.image_size[0], args.image_size[1], args.color_channels)

	# TODO relocate
	plt.figure(figsize=(10, 10))
	for images in tensor_train_dataset.take(1):
		for i in range(9):
			ax = plt.subplot(3, 3, i + 1)
			plt.imshow((images[i].numpy() + 1.0) / 2.0)
			plt.axis("off")
	plt.show(block=False)

	# TODO add as option
	latent_space = 128
	noise = tf.random.normal([16, latent_space])

	#
	generate_model = make_generator_model(latent_space, output_shape=args.image_size)
	generated_image = generate_model(noise, training=False)
	output = generate_model.output_shape
	# Create
	discriminator_model = make_discriminator_model((output[1], output[2], output[3]))

	#
	decision = discriminator_model(generated_image)
	print(decision)

	#
	generator_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
	discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)

	checkpoint_prefix = os.path.join(args.checkpoint_dir, "anime-gan")
	checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
									 discriminator_optimizer=discriminator_optimizer,
									 generator=generate_model,
									 discriminator=discriminator_model)

	seed = tf.random.normal([args.num_examples_to_generate, args.noise_dim])

	generate_model.summary()
	discriminator_model.summary()

	dcgan = DCGAN(generate_model, discriminator_model)
	dcgan.compile(generator_optimizer, discriminator_optimizer)
	history = dcgan.fit(train_images, epochs=args.epochs, batch_size=args.batch_size,
						callbacks=[
							save_images(noise=noise),
							checkpoint_callback()
						]
						)
	plotCostHistory(history.history)

	#val_acc_per_epoch = history.history['val_accuracy']
	#best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
	#print('Best epoch: %d' % (best_epoch,))

	# Save the model.
	generate_model.save(args.generate_model_filepath)
	discriminator_model.save("disc.h5")


# TODO generate and save network setup.
def GANProgram():
	parser = argparse.ArgumentParser(description='Anime GAN (Generative Adversarial Networks)')
	parser.add_argument('--epochs', type=int, default=50, dest='epochs',
						help='number of passes to train with the dataset.')
	parser.add_argument('--batch_size', type=int, default=128, dest='batch_size',
						help='number of training data per each iteration.')
	parser.add_argument('--model-file', dest='generate_model_filepath',
						default="generator-model.h5",
						help='Define fielpath to save/load model.')
	parser.add_argument('--discriminator-file', dest='discriminator_model_filepath',
						default="discriminator-model.h5",
						help='Define fielpath to save/load model.')
	parser.add_argument('--checkpoint-filepath', type=str, dest='checkpoint_dir',
						default="./training_checkpoints",
						help='Set path where check save/load model path')
	parser.add_argument('--checkpoint-every-epoch', type=str, dest='accumulate',
						default="",
						help='Define the save/load model path')
	parser.add_argument('--verbosity', type=str, dest='accumulate',
						default="",
						help='Define the save/load model path')
	parser.add_argument('--data-sets', type=str, dest='data_sets_file_paths',
						default="",
						help='Define the save/load model path')
	parser.add_argument('--data-set-directory', type=str, dest='data_sets_directory_paths',
						# action='append', nargs='*',
						help='Directory path of dataset images')
	parser.add_argument('--image-size', type=tuple, dest='image_size',
						default=(64, 64),
						help='Define the save/load model path')
	parser.add_argument('--image-filter', type=tuple, dest='accumulate',
						default="*",
						help='Define the save/load model path')
	parser.add_argument('--nr-training-data', type=int, default=3000, dest='train_set_size',
						help='Define the save/load model path')
	parser.add_argument('--seed', type=int, default=-1, dest='seed',
						help='Define the save/load model path')
	parser.add_argument('--nr_image_example_generate', type=int, default=16, dest='num_examples_to_generate',
						help='Define the save/load model path')
	parser.add_argument('--noise-dim', type=int, default=100, dest='noise_dim',
						help='Define the save/load model path')
	parser.add_argument('--color-channels', type=int, default=3, dest='color_channels',
						help='Define the save/load model path')
	parser.add_argument('--process-data-only', type=bool, default=False, dest='process_data_only',
						help='Define the save/load model path')

	args = parser.parse_args()

	logging.basicConfig(level=logging.INFO)

	if args.process_data_only:
		load_all_datasets(args)
		return

	#	for data_sets_directory_path in args.data_sets_directory_paths:
	if args.data_sets_directory_paths:
		print(args.data_sets_directory_paths)
		# data_sets_directory_path = "/mnt/ExtLaptopStore/AItemp/anime_dataset_faces/anime-face-256"
		data_dir = pathlib.Path(args.data_sets_directory_paths)
		run_train_model(args, load_from_directory(data_dir, args))

	if args.generate_model_filepath and os.path.exists(args.generate_model_filepath):
		model = tf.keras.load_model(args.generate_model_filepath)
		image = generate_image(model, args.seed)
	else:
		for path in dataset_files:
			train_X = []
			cache_path = "{0}.npz".format(pathlib.Path(path).name)
			with open(cache_path, 'rb') as f:
				train_items = np.load(f)['train']
				if len(train_X) > 0:
					train_X = np.concatenate(train_X, train_items)
				else:
					train_X = train_items  # tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(args.batch_size)
		print("number of train items {0}".format(len(train_X)))
		run_train_model(args, train_X)


if __name__ == '__main__':
	GANProgram()
