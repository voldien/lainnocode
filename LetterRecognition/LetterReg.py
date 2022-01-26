# -*- coding: utf-8 -*-
import os.path

import keras_tuner as kt
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

import procdata

print("TensorFlow version:", tf.__version__)

"""# Loading and Process of data"""

cache_path_dataX = "data_X_cache_file.npz"
cache_path_dataY = "data_Y_cache_file.npz"
if os.path.exists(cache_path_dataX):
	train_X = np.load(cache_path_dataX, allow_pickle=True)['arr_0']
	train_y = np.load(cache_path_dataY, allow_pickle=True)['arr_0']
else:
	train_X, train_y, quantive_labels = procdata.loadAllDataSet()
	train_X = train_X.astype('float32') / 255.0
	train_X = np.expand_dims(train_X, axis=-1)  # <--- add batch axis
	train_y = np.expand_dims(train_y, axis=-1)  # <--- add batch axis
	with open(cache_path_dataX, 'wb') as f:
		np.savez_compressed(f, train_X)
	with open(cache_path_dataY, 'wb') as f:
		np.savez_compressed(f, train_y)
	with open("qunative_labels") as f:
		f.write(quantive_labels)

BUFFER_SIZE = 256
BATCH_SIZE = 32
EPOCH = 48


def plotCostHistory(history, loss_label="", val_label="", title="", x_label="", y_label=""):
	for k, v in history.items():
		plt.plot(v, label=k)
	plt.title(label=title)
	plt.ylabel(ylabel=y_label)
	plt.xlabel(xlabel=x_label)
	plt.legend(loc="upper left")
	plt.show()
	plt.imsave(title + ".png")


def model_process(trainX, trainY):
	train_X, test_X, train_y, test_y = train_test_split(trainX, trainY, shuffle=True, test_size=0.25, random_state=42)

	# Neuron network input and output.
	imageShape = train_X[0].shape
	output = np.amax(train_y) + 1

	print('X_train: ' + str(train_X.shape))
	print('Y_train: ' + str(train_y.shape))
	print('X_test:  ' + str(test_X.shape))
	print('Y_test:  ' + str(test_y.shape))
	print(imageShape)

	plt.figure(figsize=(10, 4))
	plt.title("Example Data")
	nrImage = 10
	for index, (image, label) in enumerate(zip(train_X[0:nrImage], train_y[0:nrImage])):
		plt.subplot(2, 5, index + 1)
		plt.title(str.format("{0}", label))
		plt.imshow(np.reshape(image, (28, 28)), cmap=plt.cm.gray)
	plt.show()

	"""# Common Functions"""

	training_size = len(train_X)
	variance = 0.005
	varThresh = VarianceThreshold(variance)
	high_variance_training_data = train_X  # varThresh.fit(train_X)
	high_variance_training_data_size = len(high_variance_training_data)
	print(training_size)
	print(high_variance_training_data_size)

	"""# Forward Neuron Network """

	def make_forward_neuron_network(hp):
		#
		model = tf.keras.Sequential()
		model.add(layers.Flatten(input_shape=imageShape))

		model.add(layers.LeakyReLU())
		hp_rate = hp.Choice('rate', (0.1, 0.3, 0.5, 0.7, 0.9))
		model.add(layers.Dropout(rate=hp_rate))

		hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
		model.add(layers.Dense(units=hp_units, activation='relu'))

		hp_rate2 = hp.Choice('rate', (0.1, 0.3, 0.5, 0.7, 0.9))
		model.add(layers.Dropout(rate=hp_rate2))
		model.add(layers.Dense(output))

		model.summary()

		hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

		model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
					  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
					  metrics=['accuracy'])

		return model

	train_dataset = tf.data.Dataset.from_tensor_slices(high_variance_training_data).shuffle(BUFFER_SIZE).batch(
		BATCH_SIZE)

	tuner = kt.Hyperband(make_forward_neuron_network,
						 objective='val_accuracy',
						 max_epochs=EPOCH,
						 factor=3,
						 directory='my_dir',
						 project_name='intro_to_kt')

	stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
	tuner.search(train_X, train_y, epochs=50, validation_split=0.2, callbacks=[stop_early])
	best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

	fnn_model = tuner.hypermodel.build(best_hps)
	fnn_model.summary()

	checkpoint_filepath_forward_NN = os.path.join("checkpoints", "mnist_forward_neuron_network_ckpt")

	# fnn_model.load_weights(checkpoint_filepath_forward_NN)

	model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
		filepath=checkpoint_filepath_forward_NN,
		save_weights_only=True,
		monitor='accuracy',
		save_freq='epoch',
		mode='max',
		save_best_only=True)

	tf.keras.utils.plot_model(
		fnn_model, to_file='forward_model.png', show_shapes=True, show_dtype=True,
		show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96,
		layer_range=None
	)

	fnn_history = fnn_model.fit(train_X, train_y, epochs=EPOCH,
								batch_size=BATCH_SIZE, validation_split=0.2)

	val_acc_per_epoch = fnn_history.history['val_accuracy']
	best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
	print('Best epoch: %d' % (best_epoch,))

	#
	# forward_nn_history = forward_neuron_network_model.fit(high_variance_training_data, train_y, epochs=EPOCH,
	# 													  batch_size=BATCH_SIZE, callbacks=[model_checkpoint_callback])

	# Save the model for being reuse in other programs and etc.
	fnn_model.save('fnn_mnist.h5')
	fnn_model.save('fnn_mnist')

	fnn_eval_result = fnn_model.evaluate(test_X, test_y, verbose=2)
	print("[test loss, test accuracy]:", fnn_eval_result)

	plotCostHistory(fnn_history.history, title="FNN Performance History")

	"""# Convolution Neuron Network"""


def compute_cnn_model(trainX, trainY):
	train_X, test_X, train_y, test_y = train_test_split(trainX, trainY, shuffle=True, test_size=0.25,
														random_state=42)

	# Neuron network input and output.
	imageShape = train_X[0].shape
	output = np.amax(train_y) + 1

	print('X_train: ' + str(train_X.shape))
	print('Y_train: ' + str(train_y.shape))
	print('X_test:  ' + str(test_X.shape))
	print('Y_test:  ' + str(test_y.shape))
	print(imageShape)

	def make_cnn_model(hp):
		cnn_model = tf.keras.Sequential()
		#
		kernel_init = hp.Choice('kernel_initializer', ['uniform', 'lecun_uniform', 'normal', 'zero',
													   'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'])
		hp_kernel_sizes = hp.Int('filters', min_value=16, max_value=96, step=16)
		cnn_model.add(
			layers.Conv2D(hp_kernel_sizes, (3, 3), kernel_initializer=kernel_init, activation='relu', padding='same',
						  input_shape=imageShape))
		cnn_model.add(layers.MaxPooling2D((2, 2)))

		#
		cnn_model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=kernel_init, ))
		cnn_model.add(layers.MaxPooling2D((2, 2)))
		cnn_model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=kernel_init, ))

		cnn_model.add(layers.Dropout(0.3))

		cnn_model.add(layers.Dense(64, activation=tf.nn.relu, kernel_initializer=kernel_init, ))
		cnn_model.add(layers.Flatten())

		hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
		cnn_model.add(layers.Dense(units=hp_units, activation='relu', kernel_initializer=kernel_init, ))

		cnn_model.add(layers.Dense(output))

		cnn_model.summary()

		hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

		cnn_model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
						  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
						  metrics=['accuracy'])

		return cnn_model

	BUFFER_SIZE = 256
	BATCH_SIZE = 64

	loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
	optimizer = tf.keras.optimizers.Adam(1e-4)

	tuner = kt.Hyperband(make_cnn_model,
						 objective='val_accuracy',
						 max_epochs=EPOCH,
						 factor=3,
						 directory='cache',
						 project_name='cnn')

	stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
	tuner.search(train_X, train_y, epochs=50, validation_split=0.2, callbacks=[stop_early])
	best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

	cnn_model = tuner.hypermodel.build(best_hps)
	cnn_model.summary()
	cnn_model_history = cnn_model.fit(train_X, train_y, epochs=EPOCH, validation_data=(test_X, test_y),
									  validation_split=0.2)

	tf.keras.utils.plot_model(
		cnn_model, to_file='forward_model.png', show_shapes=True, show_dtype=True,
		show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96,
		layer_range=None
	)

	val_acc_per_epoch = cnn_model_history.history['val_accuracy']
	best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
	print('Best epoch: %d' % (best_epoch,))

	cnn_model.save("cnn_mnist.h5")

	cnn_model.evaluate(test_X, test_y, verbose=2)

	cnn_model.summary()

	plotCostHistory(cnn_model_history.history, title="CNN Performance History")


model_process(train_X, train_y)
compute_cnn_model(train_X, train_y)
