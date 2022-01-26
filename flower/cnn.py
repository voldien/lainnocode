import os.path

import keras_tuner as kt
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.client import device_lib
import fashion
import flower

EPOCH = 50
BUFFER_SIZE = 256
BATCH_SIZE = 64

print(device_lib.list_local_devices())


def plotCostHistory(history, loss_label="", val_label="", title="", x_label="", y_label=""):
	plt.figure()
	for k, v in history.items():
		plt.plot(v, label=k)
	plt.title(label=title)
	plt.ylabel(ylabel=y_label)
	plt.xlabel(xlabel=x_label)
	plt.legend(loc="upper left")
	plt.show()
	plt.savefig(title + ".png")


def make_cnn_model_small_images(hp, input, output):
	cnn_model = tf.keras.Sequential()
	#
	kernel_init = hp.Choice('kernel_initializer', ['uniform', 'lecun_uniform', 'normal', 'zero',
	                                               'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'])

	# TODO add stride.

	hp_kernel_filter_size_l0 = hp.Int('kernel_filter_size_0', min_value=16, max_value=96, step=16)
	hp_kernel_filter_size_l1 = hp.Int('kernel_filter_size_1', min_value=16, max_value=96, step=16)

	hp_max_pooling_size_l0 = hp.Int('max_pooling_l0', min_value=1, max_value=4, step=1)
	hp_max_pooling_size_l1 = hp.Int('max_pooling_l1', min_value=1, max_value=4, step=1)

	cnn_model.add(
		layers.Conv2D(hp_kernel_filter_size_l0, (3, 3), kernel_initializer=kernel_init, activation='relu',
		              padding='same',
		              input_shape=input))
	cnn_model.add(layers.MaxPooling2D(pool_size=hp_max_pooling_size_l0))

	#
	cnn_model.add(
		layers.Conv2D(hp_kernel_filter_size_l1, (3, 3), activation='relu', kernel_initializer=kernel_init))
	cnn_model.add(layers.MaxPooling2D(pool_size=hp_max_pooling_size_l1))

	cnn_model.add(layers.Flatten())

	hp_units = hp.Int('dense0_units', min_value=32, max_value=512, step=32)
	cnn_model.add(layers.Dense(units=hp_units, kernel_initializer=kernel_init))

	cnn_model.add(layers.Dense(output))

	cnn_model.summary()

	hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

	cnn_model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
	                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	                  metrics=['accuracy'])

	return cnn_model


def make_cnn_model_big_images(hp, input, output):
	cnn_model = tf.keras.Sequential()
	#
	kernel_init = hp.Choice('kernel_initializer', ['uniform', 'lecun_uniform', 'normal', 'zero',
	                                               'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'])

	# TODO add stride.

	hp_kernel_filter_size_l0 = hp.Int('kernel_filter_size_0', min_value=16, max_value=96, step=16)
	hp_kernel_filter_size_l1 = hp.Int('kernel_filter_size_1', min_value=16, max_value=96, step=16)
	hp_kernel_filter_size_l2 = hp.Int('kernel_filter_size_2', min_value=16, max_value=96, step=16)

	hp_max_pooling_size_l0 = hp.Int('max_pooling_l0', min_value=1, max_value=4, step=1)
	hp_max_pooling_size_l1 = hp.Int('max_pooling_l1', min_value=1, max_value=4, step=1)
	hp_max_pooling_size_l2 = hp.Int('max_pooling_l2', min_value=1, max_value=4, step=1)

	cnn_model.add(
		layers.Conv2D(hp_kernel_filter_size_l0, (3, 3), kernel_initializer=kernel_init, activation='relu',
		              padding='same',
		              input_shape=input))
	cnn_model.add(layers.MaxPooling2D(pool_size=hp_max_pooling_size_l0))

	#
	cnn_model.add(
		layers.Conv2D(hp_kernel_filter_size_l1, (3, 3), activation='relu', kernel_initializer=kernel_init))
	cnn_model.add(layers.MaxPooling2D(pool_size=hp_max_pooling_size_l1))

	cnn_model.add(
		layers.Conv2D(hp_kernel_filter_size_l2, (3, 3), activation='relu', kernel_initializer=kernel_init))
	cnn_model.add(tf.keras.layers.MaxPooling2D(pool_size=hp_max_pooling_size_l2))

	cnn_model.add(layers.Flatten())

	hp_units = hp.Int('dense0_units', min_value=32, max_value=512, step=32)
	cnn_model.add(layers.Dense(units=hp_units, kernel_initializer=kernel_init))

	cnn_model.add(layers.Dense(output))

	cnn_model.summary()

	hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

	cnn_model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
	                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	                  metrics=['accuracy'])

	return cnn_model


def compute_cnn_model(trainX, trainY, classes, dataset_name="", model_constructor=None):
	# Neuron network input and output.
	image_shape = trainX[0].shape
	num_classes_output = int(np.amax(trainY) + 1)
	print(image_shape)

	train_X, test_X, train_y, test_y = train_test_split(trainX, trainY, shuffle=True, test_size=0.25)

	print('X_train: ' + str(train_X.shape))
	print('Y_train: ' + str(train_y.shape))
	print('X_test:  ' + str(test_X.shape))
	print('Y_test:  ' + str(test_y.shape))

	#   tf.keras.layers.Rescaling(1./255),
	#   tf.keras.layers.Conv2D(32, 3, activation='relu'),
	#   tf.keras.layers.MaxPooling2D(),
	#   tf.keras.layers.Conv2D(32, 3, activation='relu'),
	#   tf.keras.layers.MaxPooling2D(),
	#   tf.keras.layers.Conv2D(32, 3, activation='relu'),
	#   tf.keras.layers.MaxPooling2D(),
	#   tf.keras.layers.Flatten(),
	#   tf.keras.layers.Dense(128, activation='relu'),
	#   tf.keras.layers.Dense(num_classes)

	def model_builder(hp):
		model = model_constructor(hp, image_shape, num_classes_output)
		return model

	tuner = kt.Hyperband(model_builder,
	                     objective='val_accuracy',
	                     max_epochs=16,
	                     factor=3,
	                     directory='cache',
	                     project_name=str.format('cnn - {0}', dataset_name))

	stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
	tuner.search(train_X, train_y, epochs=50, validation_split=0.2, callbacks=[stop_early])
	best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

	best_models = tuner.get_best_models(8)

	cnn_model = tuner.hypermodel.build(best_hps)
	cnn_model.summary()
	cnn_model_history = cnn_model.fit(train_X, train_y, epochs=EPOCH, validation_data=(test_X, test_y),
	                                  validation_split=0.2)

	val_acc_per_epoch = cnn_model_history.history['val_accuracy']
	best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
	print('Best epoch: %d' % (best_epoch,))

	cnn_model.evaluate(test_X, test_y, verbose=2)

	cnn_model.summary()

	return cnn_model, cnn_model_history


if __name__ == '__main__':

	# TODO perhaps add seperate method for constructing its own model.
	dataSets = [("flower", flower.loadDataFlower, make_cnn_model_big_images),
	            ("fashion", fashion.loadDataFashion, make_cnn_model_small_images),
	            ("cifar-100", fashion.loadDataCifar100, make_cnn_model_small_images),
	            ("cifar-10", fashion.loadDataCifar10, make_cnn_model_small_images)]
	for name, dataSet, cnn_model_builder in dataSets:
		cache_path = str.format("{0}_data_cache_file.npz", name)
		if os.path.exists(cache_path):
			train_X = np.load(cache_path, allow_pickle=True)['dataX']
			train_y = np.load(cache_path, allow_pickle=True)['dataY']
			clas = np.load(cache_path, allow_pickle=True)['classes']
		# TODO add support to write classifications as a list, CSV compa.

		else:
			train_X, train_y, clas = dataSet()
			train_X = train_X.astype('float32') / 255.0

			# Remove if dimension of 1 component
			# train_X = np.squeeze(train_X)
			# train_y = np.squeeze(train_y)
			with open(cache_path, 'wb') as f:
				np.savez_compressed(f, dataX=train_X, dataY=train_y, classes=clas)
		# with open(str.format("{0}_classifications", name), 'wb') as f:
		#	f.write(clas)
		model_path = str.format("cnn_{0}.h5", name)
		if not os.path.exists(model_path):
			model, history = compute_cnn_model(train_X, train_y, clas, name, cnn_model_builder)
			plotCostHistory(history.history, title=str.format("CNN {0} Performance History", name))
			model.save(str.format("cnn_{0}.h5", name))
		else:
			model = keras.models.load_model(model_path)

		tf.keras.utils.plot_model(
			model, to_file=str.format('cnn_{0}_model.png', name), show_shapes=True, show_dtype=True,
			show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96,
			layer_range=None
		)

		continue

		# TODO fix and improve!
		layers = model.layers
		conv_layer_index = [0, 2, 4]
		conv2DLayers = [model.layers[i].output for i in conv_layer_index]
		# for layer in layers:
		# 	if isinstance(layer, tf.keras.layers.Conv2D):
		# 		conv2DLayers.append(layer.output)

		visualModel = Model(inputs=model.inputs, outputs=conv2DLayers)
		result = visualModel.predict(train_X[0])
		fig = plt.figure()
		plt.imshow(result, cmap=plt.cm.gray)

# TODO add prediction tests.

# TODO print the best model, and result and etc.
