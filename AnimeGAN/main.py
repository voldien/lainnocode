#!/usr/bin/env python3
import os
import pickle
import time

import argparse
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from numpy import asarray
import logging

from dataProcessing import loadDatSet
from model import make_generator_model, make_discriminator_model, train_step


# tf.__version__


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


# Press the green button in the gutter to run the script.
def processImageDataset(train_images):
	# train_images = train_images.astype('float32')

	# Do per section of the
	norm1 = []
	for i in range(len(train_images)):
		norm1.append(train_images[i].astype('float32') / np.linalg.norm(train_images[i].astype('float32')))
	return np.array(norm1)
	pass


def perform_train_model(args):
	def train(dataset, seed, generate_model, args):
		for epoch in range(args.epochs):
			start = time.time()

			for image_batch in dataset:
				train_step(image_batch, args.batch_size, args.noise_dim, generate_model,
						   discriminator, generator_optimizer,
						   discriminator_optimizer, cross_entropy, args
						   )

			# Produce images for the GIF as we go
			generate_and_save_images(generate_model,
									 epoch + 1,
									 seed)

			# Save the model every 15 epochs
			if (epoch + 1) % 10 == 0:
				args.checkpoint.save(file_prefix=checkpoint_prefix)

			logging.info('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

		# Generate after the final epoch
		# display.clear_output(wait=True)
		generate_and_save_images(generate_model,
								 args.epochs,
								 seed)

	# if create model
	if args.data_sets_file_paths:
		pass

	dataset_files = ["/media/data-sets/animeface.zip", "/media/data-sets/anime-face-dataset.zip"]

	# Load training data and labels.
	(train_images, train_labels), (_, _) = loadDatSet(
		dataset_files[0:1], size=args.image_size)

	# transform the image data to float
	if args.train_set_size > 0:
		train_images = processImageDataset(train_images[0:args.train_set_size])
	else:
		train_images = processImageDataset(train_images)

	logging.info(len(train_images))

	# Extract the image size.
	BUFFER_SIZE = train_images.shape[0]
	image_size = train_images[0].shape
	assert image_size == (args.image_size[0], args.image_size[1], args.color_channels)

	train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(args.batch_size)

	noise = tf.random.normal([1, 100])

	generate_model = make_generator_model([1, 100], train_images[0].shape)
	# generator.model()

	generated_image = generate_model(noise, training=False)

	discriminator = make_discriminator_model(train_images[0].shape)
	decision = discriminator(generated_image)
	print(decision)

	cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

	#
	generator_optimizer = tf.keras.optimizers.Adam(1e-4)
	discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

	#

	checkpoint_prefix = os.path.join(args.checkpoint_dir, "ckpt")
	checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
									 discriminator_optimizer=discriminator_optimizer,
									 generator=generate_model,
									 discriminator=discriminator)

	seed = tf.random.normal([args.num_examples_to_generate, args.noise_dim])

	generate_model.summary()

	train(train_dataset, seed, generate_model, args)

	checkpoint.restore(tf.train.latest_checkpoint(args.checkpoint_dir))

	# Save the model.
	generate_model.save(args.generate_model_filepath)


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
						default=(128, 128),
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

	args = parser.parse_args()

	logging.basicConfig(level=logging.INFO)

	if args.generate_model_filepath:
		model = tf.keras.load_model(args.generate_model_filepath)
		image = generate_image(model, args.seed)
	# TODO add conditions.
	perform_train_model(args)


if __name__ == '__main__':
	animeGAN()
