import numpy
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import LeakyReLU
from keras.layers import Flatten
from tensorflow import keras
from keras.layers import Dropout
from keras.layers import LeakyReLU, ReLU
from random import Random
import keras_tuner as kt

def generate_number_set(size):
	number_Y = []
	number_X = []
	random = Random()
	for i in range(0, size):
		a = random.randint(100, 200)
		b = random.randint(100, 200)
		number_X.append([a, b, 0])
		number_Y.append(a + b)
	return numpy.asarray( number_X), np.asarray( number_Y)



def make_generator_model(hp, input, output):
	model = Sequential()

	# keras.layers.

	kernel_init = hp.Choice('kernel_initializer', ['uniform', 'lecun_uniform', 'normal', 'zero',
	                                               'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'])
	activations = hp.Choice('kernel_initializer', ['uniform', 'lecun_uniform', 'normal', 'zero',
	                                               'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'])
	hp0_units = hp.Int('dense0_units', min_value=32, max_value=512, step=32)
	hp1_units = hp.Int('dense1_units', min_value=32, max_value=512, step=32)

	# foundation for 4x4 image
	model.add(Dense(hp0_units, input_dim=input, kernel_initializer = kernel_init))
	model.add(ReLU())
	model.add(Dense(hp1_units))
	model.add(ReLU())
	model.add(Dense(output))

	hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

	model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
	                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	                  metrics=['accuracy'])

	return model



def trainModel():
	ar = 4
	input = 2
	output = 1
	EPOCH = 10
	train_X, train_y = generate_number_set(1000)

	def model_builder(hp):
		model = make_generator_model(hp, input, output)
		return model

	tuner = kt.Hyperband(model_builder,
	                     objective='val_accuracy',
	                     max_epochs=16,
	                     factor=3,
	                     directory='cache',
	                     project_name=str.format('Math'))
	stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
	tuner.search(train_X, train_y, epochs=50, validation_split=0.2, callbacks=[stop_early])
	best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

	best_models = tuner.get_best_models(8)

	cnn_model = tuner.hypermodel.build(best_hps)
	cnn_model.summary()
	cnn_model_history = cnn_model.fit(train_X, train_y, epochs=EPOCH)


if __name__ == '__main__':
	trainModel()
