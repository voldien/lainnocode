import os
import pathlib

import PIL
import numpy as np
import tensorflow as tf


def loadDataFlower():
	# def get_label(file_path):
	# 	# Convert the path to a list of path components
	# 	parts = tf.strings.split(file_path, os.path.sep)
	# 	# The second to last is the class-directory
	# 	one_hot = parts[-2] == class_names
	# 	# Integer encode the label
	# 	return tf.argmax(one_hot)
	#
	# def decode_img(img):
	# 	# Convert the compressed string to a 3D uint8 tensor
	# 	img = tf.io.decode_jpeg(img, channels=3)
	# 	# Resize the image to the desired size
	# 	return tf.image.resize(img, [img_height, img_width])
	#
	# def process_path(file_path):
	# 	label = get_label(file_path)
	# 	# Load the raw data from the file as a string
	# 	img = tf.io.read_file(file_path)
	# 	img = decode_img(img)
	# 	return img, label

	dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
	data_dir = tf.keras.utils.get_file(origin=dataset_url,
	                                   fname='flower_photos',
	                                   cache_dir='.',
	                                   untar=True)
	data_dir = pathlib.Path(data_dir)
	classes = []
	dir_images = []
	labels_batch = []
	for root, dirs, files in os.walk(data_dir):
		my_key = os.path.basename(root)

		for file_ in files:
			full_file_path = os.path.join(root, file_)
			if pathlib.Path(full_file_path).suffix in '.jpg':
				img = PIL.Image.open(full_file_path).resize((128, 128), PIL.Image.BILINEAR)
				dir_images.append(np.array(img))
				labels_batch.append(classes.index(my_key))
		for class_dir in dirs:
			classes.append(class_dir)

	assert 4 == np.amax(labels_batch)
	return np.asarray(dir_images), np.asarray(labels_batch), classes

# image_count = len(list(data_dir.glob('*/*.jpg')))
#
# batch_size = 32
# img_height = 180
# img_width = 180
#
# train_ds = tf.keras.utils.image_dataset_from_directory(
# 	data_dir,
# 	seed=123,
# 	image_size=(img_height, img_width),
# 	batch_size=batch_size)
#
#
# val_ds = tf.keras.utils.image_dataset_from_directory(
# 	data_dir,
# 	validation_split=0.2,
# 	subset="validation",
# 	seed=123,
# 	image_size=(img_height, img_width),
# 	batch_size=batch_size)
# classifications = train_ds.class_names
#
# # AUTOTUNE = tf.data.AUTOTUNE
# #
# # train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
# # val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
#
# normalization_layer = tf.keras.layers.Rescaling(1. / 255)
#
# normalized_ds = train_ds.map(lambda x, y: (x, y))
# image_batch, labels_batch = next(iter(normalized_ds))
# first_image = image_batch[0]
#
# tf.
# train_np = tfds.as_numpy(train_ds)
# test_np = tf.as_numpy(val_ds)


#
# class_names = train_ds.class_names
# print(class_names)
#
# normalization_layer = tf.keras.layers.Rescaling(1./255)
#
# normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
# image_batch, labels_batch = next(iter(normalized_ds))
# first_image = image_batch[0]
# # Notice the pixel values are now in `[0,1]`.
# print(np.min(first_image), np.max(first_image))
#
#
# AUTOTUNE = tf.data.AUTOTUNE
#
# train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
# val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
#
#
# num_classes = 5
#
# model = tf.keras.Sequential([
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
# ])
#
# #TODO add fully convolution, to see what is sees.
#
# model.compile(
#   optimizer='adam',
#   loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
#   metrics=['accuracy'])
#
#
# model.fit(
#   train_ds,
#   validation_data=val_ds,
#   epochs=3
# )
#
# list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'), shuffle=False)
# list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)
