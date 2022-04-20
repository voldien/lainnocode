import tensorflow as tf
from tensorflow import keras


class DCGAN(keras.Model):
	"""Subclass of the keras.Model class to define custom training step and loss functions"""

	# def __init__(self, latent_space_size, image_length, image_channels, build_generator, build_discriminator, **kwargs):
	#     """
	#     Parameters:
	#             seed_size: size of the random vector for the generator
	#             image_length: length of a side of the square image
	#             image_channels: number of channels in the image
	#     """
	#
	#     self.generator = build_generator(latent_space_size)
	#     self.discriminator = build_discriminator(
	#         image_length, image_channels)
	#
	#     self.__init__(self.generator, self.discriminator)

	def __init__(self, generator_model, discriminator_model, **kwargs):

		super(DCGAN, self).__init__(**kwargs)
		self.generator = generator_model
		self.discriminator = discriminator_model
		self.latent_space_size = generator_model.input_shape

		self.cross_entropy = tf.keras.losses.BinaryCrossentropy(
			from_logits=False)

	def generator_loss(self, fake_output):
		"""
		Parameters:
				fake_output: Tensor containing the respective discriminator's predictions for the batch of images produced
				by generator (fake iamges).

		Returns:
				cross entropy loss between labels for real images (1's) and the discriminator's estimate
		"""

		# The objective is to penalize the generator whenever it produces images which the discriminator classifies as 'fake'
		return self.cross_entropy(tf.ones_like(fake_output), fake_output)

		# smooth parameter is used to induce one sided label smoothing. It can be tuned accordingly
	def discriminator_loss(self, real_output, fake_output, smooth=0.1):
		"""
		Parameters:
				real_output: Tensor containing the respective discriminator's predictions for the batch of images taken from
										the dataset (real images).
				fake_output: Tensor containing the respective discriminator's predictions for the batch of images produced
										by generator (fake images).

		Returns:
				total_loss: Loss of the discriminator for misclassifying images
		"""
		# label for real image is (1-smooth)
		real_loss = self.cross_entropy(tf.ones_like(
			real_output)*(1-smooth), real_output)
		fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
		total_loss = real_loss + fake_loss
		return total_loss

	def compile(self, generator_optimizer, discriminator_optimizer):
		"""
		configures model for training by adding optimizers

		Parameters:
				generator_optimizer: keras optimizer to be used to train generator.
				discriminator_optimizer: keras optimizer to be used to train discriminator.
		"""
		super(DCGAN, self).compile()
		self.generator_optimizer = generator_optimizer
		self.discriminator_optimizer = discriminator_optimizer

	@tf.function
	def train_step(self, data):

		batch_size = tf.shape(data)[0]
		#TODO resolve
		batch_size = 128
		self.latent_space_size = 128

		# feed a random input to generator
		seed = tf.random.normal(shape=(batch_size, self.latent_space_size))

		with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

			# generate image using generator
			generated_image = self.generator(seed, training=True)

			# discriminator's prediction for real image
			real_output = self.discriminator(data, training=True)

			# discriminator's estimate for fake image
			fake_output = self.discriminator(
				generated_image, training=True)

			# compute loss
			gen_loss = self.generator_loss(fake_output)
			disc_loss = self.discriminator_loss(real_output, fake_output)

			# optimize generator first
			generator_grad = gen_tape.gradient(
				gen_loss, self.generator.trainable_variables)
			discriminator_grad = disc_tape.gradient(
				disc_loss, self.discriminator.trainable_variables)

			# optimize discriminator after generator
			self.generator_optimizer.apply_gradients(
				zip(generator_grad, self.generator.trainable_variables))
			self.discriminator_optimizer.apply_gradients(
				zip(discriminator_grad, self.discriminator.trainable_variables))

		return {
			"generator loss": gen_loss,
			"discriminator_loss": disc_loss
		}
