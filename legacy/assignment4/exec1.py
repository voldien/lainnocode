import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs

from KBisMean import bkmeans, compute_mean


def test():
	def assertShape(sample_size, clusters):
		assert clusters.shape[0] == sample_size

	# Testing values.
	k = [2, 3, 4, 5, 6, 7]
	d = 2
	n_samples = 150
	iter = [5, 15, 25]
	plotFigureWidth = 4

	results = []
	kClusterResults = []
	for it in iter:
		kResult = []
		X, y = make_blobs(n_samples=n_samples, centers=d, n_features=d, random_state=True)
		for _k in k:
			print("k", _k, "iter", it)
			# Perform bisecting clustering.
			labels = bkmeans(X, k=_k, iter=it)
			assertShape(n_samples, labels)

			# Computes the centeriods.
			centers = []
			for _kK in range(0, _k):
				_cluster = [X[j] for j, i in enumerate(labels) if i == _kK]
				centers.append(compute_mean(_cluster))

			# Store the results.
			results.append([X, y, labels, centers, (d, _k, it)])
			kResult.append([X, y, labels, centers, (d, _k, it)])
		kClusterResults.append(kResult)

	# Plot the result.
	plt.figure(figsize=(40, 40))
	w = int((len(results) / plotFigureWidth)) + 2
	h = int(len(results) % plotFigureWidth) + 2
	plt.title("Bisecting KMean Clustering")
	for i, r in enumerate(results):
		_x, _y, cl, centers, att = r
		plti = plt.subplot(w, h, i + 1)

		plti.scatter(_x[:, 0], _x[:, 1], c=cl)
		plti.set_title("Dims {}, K {}, iter {}".format(att[0], att[1], att[2]))
		for center in centers:
			plti.scatter(center[0], center[1], c='r')

	plt.show()
	return


	# Elbow
	# plt.figure(figsize=(40, 5))
	# for i, kResult in enumerate(kClusterResults):
	# 	plti = plt.subplot(1, len(kClusterResults), i + 1)
	# 	Xline = []
	# 	YLine = []
	# 	for _kv, kv in enumerate(kResult[0]):
	# 		X, y, labels, att = kv
	# 		eSum = 0
	# 		for j, _i in enumerate(labels):
	# 			esum = 0
	# 		Xline.append(_kv + 1)
	# 		Xline.append(esum)
	#
	# 	plti.plot(Xline, YLine)
	# plt.show()

if __name__ == '__main__':
	test()
