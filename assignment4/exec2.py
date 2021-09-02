import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


def assertShape(sa, n):
	assert sa.shape[1] == 2 and sa.shape[0] == n


def sammon(X, max_iter, e=1e-4, rate=0.01):
	"""
	E = 1 / sum(X distances) * sum( (X distances - Y distances)^2 / X distances)
	DE/Dy = (-2 / c) * sum [i = 1 j != p, ( X distance - Y Distance / X distance * Y distances)] ( y - y )
	c = sum(y distance i,j) i < j
	:param X: Extended matrix.
	:param max_iter:
	:param e:
	:param rate:
	:return:  n × 2
	"""
	n = 2  # Dimensions.
	distX = cdist(X, X, 'euclidean')

	y = np.random.normal(0.0, 1.0, [X.shape[0], n])

	for it in range(max_iter):
		distY = cdist(y, y, 'euclidean')
		c = distX.sum() / 2.0

		# Compute E.
		e0 = 1.0 / c
		e1 = (distX - distY) ** 2.0
		e2 = distX
		# Divide while prevent divide by zero.
		E = e0 * np.divide(e1, e2, out=np.zeros_like(e1), where=e1 != 0).sum() / 2.0

		print("E:", E)

		if E < e:
			return y

		# Prevent C from getting to small.
		if abs(c) < 0.0001:
			c = 0.01 * np.sign(c)
		coff = (-2.0 / c)
		for i in range(y.shape[0]):
			delta_0 = np.zeros(y.shape[1])
			delta_1 = np.zeros(y.shape[1])

			for j in range(y.shape[0]):
				if i == j:
					continue

				# Compute shared variables.
				dpi_x = distX[i, j]
				dpi_y = distY[i, j]

				# Handle small denominator
				if abs(dpi_x) < 0.0001:
					dpi_x = 0.01 * np.sign(dpi_x)
				if abs(dpi_y) < 0.0001:
					dpi_y = 0.01 * np.sign(dpi_y)

				dpi_diff = (dpi_x - dpi_y)
				dpi_mul_inv = 1.0 / (dpi_x * dpi_y)
				ydiff = (y[i] - y[j])
				ydiffexp2 = ydiff ** 2

				# First derivative
				delta_0 = delta_0 + dpi_diff * dpi_mul_inv * ydiff

				# Second derivative
				delta1_v = (dpi_diff - ((ydiffexp2 / dpi_y) * (1 + (dpi_diff / dpi_y))))
				delta_1 = delta_1 + dpi_mul_inv * delta1_v

			delta_0 = coff * delta_0
			delta_1 = coff * delta_1

			delta = delta_0 / np.absolute(delta_1)
			y[i] = y[i] - rate * delta

	if it == max_iter - 1:
		print("Warning: max_iter exceeded. Sammon mapping may not have converged...")

	assertShape(y, len(X))

	return y
