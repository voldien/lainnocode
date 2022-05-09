from collections import Counter

import numpy as np


def euclidean(a, b, k=1):
	return np.linalg.norm(np.array(a) - np.array(b))


def manhattan(a, b, k=1):
	b = np.abs(np.array(a) - np.array(b))
	return np.sum(b)


def minkowski(a, b, k=1):
	q = k
	qinv = 1.0 / q
	b = np.abs(np.array(a) - np.array(b))
	b = b ** q
	return np.sum(b) * qinv


def KNNClassifier(X, Y, predict, k=3, distance=euclidean):
	votes = knnVoteList(X, Y, predict, k, distance)

	# Get most common label.
	vote_result = Counter(votes).most_common(1)[0][0]
	return vote_result


def knnVoteList(X, Y, predict, k, distance=euclidean):
	if k < 1:
		raise ValueError("K must be greater or equal to 1")

	distances = []
	i = 0

	# Compute all distance between predict and all cluster points.
	for point in X:
		dist = distance(point, predict, k)
		group = Y[i]
		distances.append([dist, group])
		i = i + 1

	# Sort based on the distances.
	sort = sorted(distances)
	votes = [j[1] for j in sort[:k]]

	# Get most common label.
	return votes


def knnCandidates(X, Y, predict, k, distance=euclidean):
	if k < 1:
		raise ValueError("K must be greater or equal to 1")

	distances = []
	i = 0

	# Compute all distance between predict and all cluster points.
	for point in X:
		dist = distance(point, predict, k)
		group = Y[i]
		distances.append([dist, group, point])
		i = i + 1

	# Sort based on the distances.
	sort = sorted(distances)
	votes = [j[1] for j in sort[:k]]
	x = [j[2] for j in sort[:k]]

	# Get most common label.
	return [[x_[0], y_] for x_, y_ in zip(x, votes)]


def classifyArray(X, y, k, xy_mesh, distance=euclidean):
	arra = []
	for point in xy_mesh:
		c = knnVoteList(X, y, point, k, distance)
		arra.append(c[0])
	return np.array(arra)


def meanDistance(predict, points):

	if len(points) <= 0:
		raise ValueError("Must exist atleast a single point")
	if len(points) == 1:
		return points[0]
	pn = np.array(points[0])

	for p in points[1:]:
		pn = pn + (1.0 / 2.0) * (np.array(p) - pn)
	return pn
