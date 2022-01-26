import io
import os.path
import zipfile

import numpy as np
from keras.datasets import mnist

k49_mnist = "/media/data-sets/kanji-mnist.zip"


def loadK49():
	sets = []

	file_set = ["k49-train-imgs.npz", "k49-train-labels.npz", "k49-test-imgs.npz", "k49-test-labels.npz"]
	with zipfile.ZipFile(k49_mnist, 'r') as zip:
		for i, path in enumerate(file_set):
			data = zip.read(path)
			with zipfile.ZipFile(io.BytesIO(data), 'r') as f:
				data_ = f.read(f.namelist()[0])
				sets.append(np.load(io.BytesIO(data_), allow_pickle=True))
	return (sets[0], sets[1]), (sets[2], sets[3])


kanji_mnist = "/media/data-sets/kanji-mnist.zip"


def loadKanjiMNIST():
	sets = []
	file_set = ["kmnist-train-imgs.npz", "kmnist-train-labels.npz", "kmnist-test-imgs.npz", "kmnist-test-labels.npz"]
	with zipfile.ZipFile(kanji_mnist, 'r') as zip:
		for i, path in enumerate(file_set):
			data = zip.read(path)
			with zipfile.ZipFile(io.BytesIO(data), 'r') as f:
				data_ = f.read(f.namelist()[0])
				sets.append(np.load(io.BytesIO(data_), allow_pickle=True))
	return (sets[0], sets[1]), (sets[2], sets[3])

math_symbols = "/media/data-sets/math_symbols.zip"
def loadMathSymbols():
	with zipfile.ZipFile(math_symbols, 'r') as zip:
		dirs = list(set([os.path.dirname(x) for x in zip.namelist()]))
		for dir in dir:
			pass
		# for i, path in enumerate(file_set):
		# 	pass
	return ()

english_symbols = "/media/data-sets/english_letters.zip"
def loadEnglishSymbols():
	pass


def loadAllDataSet():
	(mnist_train_X, mnist_train_y), (mnist_test_X, mnist_test_y) = mnist.load_data()
	mnist_qu = [str(i) for i in range(0, 10)]
	(k49_mnist_train_X, k49_mnist_train_y), (k49_mnist_test_X, k49_mnist_test_y) = loadK49()
	(Kanji_mnist_train_X, Kanji_mnist_train_y), (Kanji_mnist_test_X, Kanji_mnist_test_y) = loadKanjiMNIST()

	quantive_labels = []

	labelSets = []
	labelSets.append(np.concatenate((mnist_train_y, mnist_test_y)))
	labelSets.append(np.concatenate((k49_mnist_train_y, k49_mnist_test_y)))
	labelSets.append(np.concatenate((Kanji_mnist_train_y, Kanji_mnist_test_y)))

	labels = []
	nrLabels = 0
	for labelset in labelSets:
		for label in labelset:
			labels.append(label + nrLabels)
		nrLabels += np.amax(labelset)

	return np.concatenate((mnist_train_X, mnist_test_X, k49_mnist_train_X, k49_mnist_test_X, Kanji_mnist_train_X,
						   Kanji_mnist_test_X)), np.array(labels), quantive_labels
