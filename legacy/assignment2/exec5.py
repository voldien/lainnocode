import gzip
import struct
from array import array as pyarray

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

from regression import multiLogisticFit, multiLogisticPredict, logicError

samplesSize = [100, 100, 150, 500, 1000, 2000, 3000, 4000, 6000, 8000, 10000, 40000, 60000]
testSize = [500, 600, 700, 800, 900, 1000, 3000, 4000, 6000, 8000, 9000, 10000, 10000]
digits = np.arange(10)


def loadMNIST(imagePath, labelPath, size=1000, digits=np.arange(10)):
	"""

	:param imagePath:
	:param labelPath:
	:param size:
	:param digits:
	:return:
	"""
	N = size

	with gzip.open(labelPath, 'rb') as finf:
		magic_nr, size = struct.unpack(">II", finf.read(8))
		lbl = pyarray("b", finf.read())

		ind = [k for k in range(size) if lbl[k] in digits]
		labels = np.zeros((N, 1), dtype=np.int8)
		for i in range(N):
			labels[i] = lbl[ind[i]]
		finf.close()

	with gzip.open(imagePath, 'rb') as fimg:
		magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
		img = pyarray("B", fimg.read())

		ind = [k for k in range(size) if lbl[k] in digits]
		images = np.zeros((N, rows * cols), dtype=np.float)

		for i in range(N):  # int(len(ind) * size/100.)):
			images[i] = np.array(img[ind[i] * rows * cols: (ind[i] + 1) * rows * cols]) \
							.reshape((rows * cols)) / 255.0

		fimg.close()

	labels = [label[0] for label in labels]
	return images, labels


MNISTImgTrain = "datasets/train-images-idx3-ubyte.gz"
MNISTLabelTrain = "datasets/train-labels-idx1-ubyte.gz"
MNISTImg = "datasets/t10k-images-idx3-ubyte.gz"
MNISTLabel = "datasets/t10k-labels-idx1-ubyte.gz"

# Test if the images are loaded correctly!
train_img, train_lbl = loadMNIST(MNISTImgTrain, MNISTLabelTrain, 5)

plt.figure(figsize=(20, 4))
for index, (image, label) in enumerate(zip(train_img[0:5], train_lbl[0:5])):
	plt.subplot(1, 5, index + 1)
	plt.imshow(np.reshape(image, (28, 28)), cmap=plt.cm.gray)
	plt.title('Training: %i\n' % label, fontsize=20)
plt.show()

# 2 Regression model Logistic
for s, t in zip(samplesSize[::-1], testSize):
	X, y = loadMNIST(MNISTImgTrain, MNISTLabelTrain, 60000)
	tX, ty = loadMNIST(MNISTImg, MNISTLabel, t)
	logisticRegr = LogisticRegression(solver='lbfgs', verbose=1, n_jobs=4, max_iter=1000)

	logisticRegr.fit(X, y)
	predictions = logisticRegr.predict(tX)
	accuracy = logisticRegr.score(tX, ty)

	#
	y_pred_proba = logisticRegr.predict_proba(tX)
	fpr, tpr, _ = metrics.roc_curve(ty, y_pred_proba)

	cm = metrics.confusion_matrix(ty, predictions)
	print(cm)
	print("sklearn-LogisticRegression::" "train:", len(X), "test:", len(ty), "accuracy:", accuracy / len(ty))

# 2 Regression model Logistic from previous assignments
beta = np.zeros(28 * 28 + 1)

predictions = []
for s, t in zip(samplesSize, testSize):
	X, y = loadMNIST(MNISTImgTrain, MNISTLabelTrain, t)
	tX, ty = loadMNIST(MNISTImg, MNISTLabel, s)

	# Compute the best
	w, b = multiLogisticFit(X, y, 10)

	predicted = multiLogisticPredict(tX, b, w)
	score = logicError(predicted, ty)

	print("Logistic-Regression::" "train:", len(X), "test:", len(ty), "accuracy:", score)
# cm = metrics.confusion_matrix(ty, predicted)
# print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
# print("Precision:", metrics.precision_score(y_test, y_pred))
# print("Recall:", metrics.recall_score(y_test, y_pred))
