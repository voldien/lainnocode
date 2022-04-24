import argparse
import tensorflow as tf
from random import randrange

from matplotlib import pyplot as plt
from numpy import asarray

from util.image import generate_image

parser = argparse.ArgumentParser(description='GAN (Generative Adversarial Networks) Image Generator')
parser.add_argument('--model-file', dest='generate_model_filepath',
					default="generator-model.h5",
					help='Define fielpath to save/load model.')

parser.add_argument('--verbosity', type=str, dest='accumulate',
					default="",
					help='Define the save/load model path')

parser.add_argument('--image-size', type=tuple, dest='image_size',
					default=(64, 64),
					help='Define the save/load model path')

parser.add_argument('--image-filter', type=tuple, dest='accumulate',
					default="*",
					help='Define the save/load model path')

parser.add_argument('--seed', type=int, default=randrange(10000000), dest='seed',
					help='Define the save/load model path')

args = parser.parse_args()

model = tf.keras.models.load_model(args.generate_model_filepath)

latent_space_size = model.input_shape
print(latent_space_size)

latent_point = tf.random.normal([16, latent_space_size[1]], seed=args.seed)


def generate_and_save_images(model, epoch, seed):
	# Notice `training` is set to False.
	# This is so all layers run in inference mode (batchnorm).
	# Generate the image.
	predictions = generate_image(model, seed)

	# Generate the figure frame.
	fig = plt.figure(figsize=(4, 4))
	#
	for i in range(predictions.shape[0]):
		plt.subplot(4, 4, i + 1)
		rgb = (predictions[i, :, :, 0:3] + 1.0) / 2.0
		plt.imshow(asarray(rgb))
		plt.axis('off')

	fig.savefig('image_at_epoch_{:04d}.png'.format(epoch))
	plt.show(block=True)


generate_and_save_images(model, 100, latent_point)
# image.show()
