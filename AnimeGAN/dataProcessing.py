import zipfile
from io import StringIO, BytesIO  ## for Python 3

import PIL.Image
import numpy as np
import PIL
from numpy import asarray

import concurrent.futures as cf


#TODO make the filter a lamba method
def loadImageDataSubSet(path, subset, resize=(128, 128), filter=(".jpg", ".JPG", ".png", ".png")):
	images = []
	_n = int(len(subset))
	with zipfile.ZipFile(path, 'r') as zip:
		for i in range(_n):
			file_in_zip = subset[i]
			if (".jpg" in file_in_zip or ".JPG" in file_in_zip or ".png" in file_in_zip):
				data = zip.read(file_in_zip)
				stream = BytesIO(data)
				image = PIL.Image.open(stream)
				image = image.resize(resize, PIL.Image.BILINEAR)
				images.append(asarray(image))
				stream.close()
	return images


def load_image_data(pool, path, size):
	future_to_image = []
	with zipfile.ZipFile(path, 'r') as zip:
		zlist = zip.namelist()
		nr_chunks = 32
		chunk_size = int(len(zlist) / nr_chunks)
		for i in range(nr_chunks):
			subset = zlist[chunk_size * i: chunk_size * (i + 1)]
			task = pool.submit(loadImageDataSubSet, path, subset)
			future_to_image.append(task)
	return future_to_image


def loadDatSet(paths, filter=None, ProcessOverride=None, size=(128, 128)):
	future_to_image = []
	total_data = []
	with cf.ProcessPoolExecutor() as pool:
		for path in paths:
			for f in load_image_data(pool, path, size):
				future_to_image.append(f)
		for future in cf.as_completed(set(future_to_image)):
			try:
				data = future.result()
				for x in data:
					total_data.append(x)
			except Exception as exc:
				print('%r generated an exception: %s' % ("url", exc))
			else:
				print(str.format('{0} page is {1} bytes', "url", len(data)))
			del data
	return (np.array(total_data), None), (None, None)
