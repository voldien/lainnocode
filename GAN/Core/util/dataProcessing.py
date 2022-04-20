import pathlib
import zipfile
from io import StringIO, BytesIO  ## for Python 3
import imghdr

import PIL.Image
import numpy as np
import PIL
from numpy import asarray

import concurrent.futures as cf


# Press the green button in the gutter to run the script.
# TODO
def processImageDataset(train_images):
	# Do per section of the
	for i, image in enumerate(train_images.astype('float32')):
		train_images[i] = train_images[i] / 255.0
	return train_images


# image_count = len(list(data_dir.glob('*/*.jpg')))
# print(image_count)

# roses = list(data_dir.glob('roses/*'))
# PIL.Image.open(str(roses[1]))


# TODO make the filter a lamba method
def loadImageDataSubSet(path, subset, resize=(128, 128), filter=(".jpg", ".JPG", ".png", ".png")):
	images = []
	_n = int(len(subset))
	with zipfile.ZipFile(path, 'r') as zip:
		for i in range(_n):
			file_in_zip = subset[i]

			if pathlib.Path(file_in_zip).suffix in filter:
				try:
					data = zip.read(file_in_zip)
					stream = BytesIO(data)
					if imghdr.what(stream) is not None:
						image = PIL.Image.open(stream)
						image = image.resize(resize, PIL.Image.BILINEAR)
						images.append(np.asarray(image))
					stream.close()
				except Exception as exc:
					print('{0} generated an exception: {1}'.format(file_in_zip, exc))
	return images


def load_image_data(pool, path, size):
	future_to_image = []
	with zipfile.ZipFile(path, 'r') as zip:
		zlist = zip.namelist()
		nr_chunks = 32
		chunk_size = int(len(zlist) / nr_chunks)
		for i in range(nr_chunks):
			subset = zlist[chunk_size * i: chunk_size * (i + 1)]
			task = pool.submit(loadImageDataSubSet, path, subset, size)
			future_to_image.append(task)
	return future_to_image


def loadImagedataSet(path, filter=None, ProcessOverride=None, size=(128, 128)):
	future_to_image = []
	total_data = []
	with cf.ProcessPoolExecutor() as pool:
		for f in load_image_data(pool, path, size):
			future_to_image.append(f)
			for future in cf.as_completed(set(future_to_image)):
				try:
					data = future.result()
					for x in data:
						total_data.append(x)
				except Exception as exc:
					print('{0} generated an exception: {1}'.format("url", exc))
				del data
	return (np.asarray(total_data), None), (None, None)


def loadImageDataSet(paths, filter=None, ProcessOverride=None, size=(128, 128)):
	image_sets = []
	for path in paths:
		(image, _0), (_1, _2) = loadImagedataSet(path, filter, ProcessOverride, size)
		image_sets.append(image)
	return (np.concatenate((x for x in image_sets)), None), (None, None)
