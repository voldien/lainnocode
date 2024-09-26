import numpy as np
from sklearn.cluster import KMeans


def compute_mean(X):
	"""

	:param predict:
	:param points:
	:return:
	"""
	npX = np.array(X)
	length = npX.shape[0]
	sum_x = np.sum(npX[:, 0])
	sum_y = np.sum(npX[:, 1])
	return [sum_x / length, sum_y / length]


def sse(X, C):
	_c = np.array(C)
	m = (1.0 / len(X))
	return m * np.sum(np.array([np.abs(x - _c) * np.abs(x - _c) for x in X]))


def bkmeans(X, k=2, iter=10):
	"""

	:param X:
	:param y:
	:return: array of indices.
	"""
	clusters = []
	minSSE = np.inf
	labels = None
	centroid = None

	for it in range(iter):
		kmean = KMeans(n_clusters=2, n_jobs=-1, random_state=True)
		kmean.fit(X)
		SSE = kmean.inertia_
		if SSE < minSSE:
			minSSE = SSE
			labels = kmean.labels_
			centroid = kmean.cluster_centers_

	assert len(labels) == len(X)

	xl = [i for i, l in enumerate(labels) if l == 0]  # X indices
	x_l = [X[l] for l in xl]  # Cached X
	_sse = sse(np.array(x_l), centroid[0])
	clusters.append([_sse, centroid[0], xl, x_l, 0])

	xr = [i for i, l in enumerate(labels) if l == 1]  # X indices
	x_r = [X[l] for l in xr]  # Cached X
	_sse = sse(x_r, centroid[1])
	clusters.append([_sse, centroid[1], xr, x_r, 1])

	while len(clusters) < k:
		maxIndex = 0
		maxSSE = -1
		# Choose the cluster with Maximum SSE to split
		for j in range(len(clusters)):
			SSE = clusters[j][0]
			if SSE > maxSSE:
				maxIndex = j
				maxSSE = SSE

		minSSE = np.inf
		labels = None
		centroid = None

		# Select the cluster to split.
		selectedSplitCluster = clusters[maxIndex]
		pointsInCluster = selectedSplitCluster[3]
		for it in range(iter):
			kmean = KMeans(n_clusters=2, n_jobs=-1, random_state=True)
			kmean.fit(pointsInCluster)
			SSE = kmean.inertia_
			if SSE < minSSE:
				minSSE = SSE
				labels = kmean.labels_
				centroid = kmean.cluster_centers_

		clusterIndex = selectedSplitCluster[2]

		# Update the index on the left side.
		xl = [i for i, l in zip(clusterIndex, labels) if l == 0]  # X indices
		# Update cached feature variables.
		x_l = [X[l] for l in xl]
		_sse = sse(x_l, centroid[0])
		clusters[maxIndex] = [_sse, centroid[0], xl, x_l, maxIndex]

		# Update the index
		xr = [i for i, l in zip(clusterIndex, labels) if l == 1]  # X indices
		# Update cached feature variables.
		x_r = [X[l] for l in xr]
		_sse = sse(x_r, centroid[1])
		clusters.append([_sse, centroid[1], xr, x_r, maxIndex + 1])

	assert len(clusters) == k

	indices = np.zeros(len(X), dtype=np.int)
	for j, cluster in enumerate(clusters):
		x_indices = cluster[2]
		for l, i in enumerate(x_indices):
			indices[i] = int(j)

	assert len(indices) == len(X)
	return indices
