#!/usr/bin/env python3
import argparse
import logging
import os
import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from numpy import asarray

from model import make_generator_model, make_discriminator_model, train_step
from util.dataProcessing import loadImagedataSet

# TODO remove
#dataset_files = ["/media/data-sets/anime/animeface.zip"]#, "/media/data-sets/anime/anime-face-dataset.zip",
                 #"/media/data-sets/anime/animefacedataset.zip"]
dataset_files = ["/media/data-sets/anime/anime-face-dataset.zip"]
# "/media/data-sets/anime/moeimouto-faces.zip"]


def load_all_datasets(args):
	# dataset_files = args.data_sets_file_paths
	for path in dataset_files:
		cache_path = "{0}.npz".format(pathlib.Path(path).name)
		if not os.path.exists(cache_path):
			print("loading dataset archive {0}".format(path))
			(train_images, train_labels), (_, _) = loadImagedataSet(path, size=args.image_size)

			#
			if train_images.shape[-1] > 3:
				train_images = np.expand_dims(train_images, axis=-1)  # <--- add batch axis

			train_images = train_images.astype('float32')
			train_images *= (1.0 / 255.0)


			with open(cache_path, 'wb') as f:
				np.savez_compressed(f, train=train_images)
			del train_images


def generate_image(model, seed):
	return model(seed, training=False)


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
		rgb = predictions[i, :, :, 0:3] * 127.5 + 127.5
		rgb /= 255.0
		plt.imshow(asarray(rgb))
		plt.axis('off')

	plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))


def perform_train_model(args, train_images):
	#

	# if create model
	if args.data_sets_file_paths:
		pass

	logging.info(len(train_images))

	# Extract the image size.
	BATCH_SIZE = 256
	BUFFER_SIZE = train_images.shape[0]
	image_size = train_images[0].shape
	assert image_size == (args.image_size[0], args.image_size[1], args.color_channels)

	tensor_train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(args.batch_size)

	noise = tf.random.normal([1, 100])
	generate_model = make_generator_model([1, 100], (100,))

	generated_image = generate_model(noise, training=False)
	output = generate_model.output_shape
	discriminator = make_discriminator_model((output[1], output[2], output[3]))
	decision = discriminator(generated_image)
	print(decision)

	cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

	#
	generator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5)
	discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5)

	checkpoint_prefix = os.path.join(args.checkpoint_dir, "anime-gan")
	checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
	                                 discriminator_optimizer=discriminator_optimizer,
	                                 generator=generate_model,
	                                 discriminator=discriminator)

	seed = tf.random.normal([args.num_examples_to_generate, args.noise_dim])

	generate_model.summary()

	def train(tensor_train_dataset, seed, generate_model, args):

		checkpoint.restore(tf.train.latest_checkpoint(args.checkpoint_dir))

		for epoch in range(args.epochs):
			start = time.time()

			for i, image_batch in enumerate(tensor_train_dataset):
				batch_start_time = time.time()
				train_step(image_batch, generate_model,
				           discriminator, generator_optimizer,
				           discriminator_optimizer, cross_entropy)
				print("{0}/{1} {2} sec'".format(i * BATCH_SIZE, len(train_images), (time.time() - batch_start_time)),
				      end='\r')

			# Produce images for the GIF as we go
			generate_and_save_images(generate_model,
			                         epoch + 1,
			                         seed)

			# Save the model every 15 epochs
			if (epoch + 1) % 5 == 0:
				args.checkpoint.save(file_prefix=checkpoint_prefix)

			logging.info('Time for epoch {} is {} sec\n'.format(epoch + 1, time.time() - start))

		# Generate after the final epoch
		# display.clear_output(wait=True)
		generate_and_save_images(generate_model,
		                         args.epochs,
		                         seed)

	train(tensor_train_dataset, seed, generate_model, args)

	# Save the model.
	generate_model.save(args.generate_model_filepath)
	decision.save("disc.h5")


def perform_generate_images(generate_model):
	pass


def animeGAN():
	parser = argparse.ArgumentParser(description='Anime GAN (Generative Adversarial Networks)')
	parser.add_argument('--epochs', type=int, default=50, dest='epochs',
	                    help='an integer for the accumulator')
	parser.add_argument('--batch_size', type=int, default=256, dest='batch_size',
	                    help='an integer for the accumulator')
	parser.add_argument('--model-file', dest='generate_model_filepath',
	                    default="animeGAN.model",
	                    help='Define the save/load model path')
	parser.add_argument('--checkpoint-filepath', type=str, dest='checkpoint_dir',
	                    default="./training_checkpoints",
	                    help='Define the save/load model path')
	parser.add_argument('--checkpoint-every-epoch', type=str, dest='accumulate',
	                    default="",
	                    help='Define the save/load model path')
	parser.add_argument('--verbosity', type=str, dest='accumulate',
	                    default="",
	                    help='Define the save/load model path')
	parser.add_argument('--data-sets', type=str, dest='data_sets_file_paths',
	                    default="",
	                    help='Define the save/load model path')
	parser.add_argument('--image-size', type=tuple, dest='image_size',
	                    default=(32, 32),
	                    help='Define the save/load model path')
	parser.add_argument('--image-filter', type=tuple, dest='accumulate',
	                    default="",
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
					train_X = train_items
		print("number of train items {0}".format(len(train_X)))
		perform_train_model(args, train_X)


if __name__ == '__main__':
	animeGAN()
