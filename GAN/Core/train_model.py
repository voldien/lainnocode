import GAN

import keras.layers
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LeakyReLU
from keras.layers import Normalization
# tf.__version__
from keras.optimizer_v2.adam import Adam

from tensorflow.keras import layers


def make_generator_model(noise_dim, input_shape, output_shape=(1, 1)):
	model = tf.keras.Sequential()

    init = tf.keras.initializers.TruncatedNormal(stddev=0.02)

    n2 = math.log2(float(IMAGE_SIZE[0]))
    num_layers = max(int(n2) - 4, 0)

    # foundation for 4x4 image
    n_nodes = 1024 * 4 * 4
    model.add(layers.Dense(
        n_nodes, kernel_initializer=init, input_dim=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Reshape((4, 4, 1024)))
    # Note: None is the batch size
    assert model.output_shape == (None, 4, 4, 1024)

    # upsample to 8x8
    model.add(layers.Conv2DTranspose(filters=1024, kernel_size=(4, 4), strides=(2, 2),
                                     use_bias=False, padding='same', kernel_initializer=init))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    for i in range(0, num_layers + 1):
        filter_size = int(512 / (2 ** i))
        image_size = (16 * (2 ** i), 16 * (2 ** i))

        model.add(layers.Conv2DTranspose(filters=filter_size, kernel_size=(4, 4), strides=(2, 2), use_bias=False, padding='same',
                                         kernel_initializer=init))
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())
        assert model.output_shape == (
            None, image_size[0], image_size[1], filter_size)

    # output layer
    model.add(layers.Conv2DTranspose(filters=3, kernel_size=(4, 4), strides=(
        1, 1), padding='same', kernel_initializer=init))
    model.add(layers.Activation('tanh'))
    # TODO add color channel
    assert model.output_shape == (None, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    return model


def make_discriminator_model(input_shape):
    model = tf.keras.Sequential()

    n_layers = max(int(math.log2(IMAGE_SIZE[1])) - 3, 0)

    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), use_bias=False, padding='same',
                            input_shape=[IMAGE_SIZE[0], IMAGE_SIZE[1], 3]))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    for i in range(0, n_layers):
        filter_size = 128 * (2 ** i)
        kernel_size = (5, 5)
        model.add(layers.Conv2D(filter_size, kernel_size=kernel_size,
                                strides=(2, 2), use_bias=False, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(1, kernel_size=4, strides=(
        2, 2), padding='valid', use_bias=False))
    model.add(layers.Flatten())
    model.add(layers.Activation('sigmoid'))

    return model