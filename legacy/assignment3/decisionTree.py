from matplotlib import style
import numpy as np
from matplotlib.colors import ListedColormap


def mesh2Dgrid(min, max, resolution):
	if resolution < 0:
		raise ValueError("Resolution of the grid must be greater than 0.")
	# Max and min of the bounding world.
	x_min, x_max = min[0] - 0.1, max[0] + 0.1
	y_min, y_max = min[1] - 0.1, max[1] + 0.1

	if x_min > x_max or y_min > y_max:
		raise ValueError("Min can not be greater than max.")
	#
	xw = np.arange(x_min, x_max, resolution)
	yh = np.arange(y_min, y_max, resolution)
	xx, yy = np.meshgrid(xw, yh)  # Mesh Grid

	x1, x2 = xx.ravel(), yy.ravel()  # Turn to two Nx1 arrays
	assert len(xx) > 0 and len(yy) > 0
	return xx, yy, x1, x2


def drawDecisionBoundary(explt, model, max, min, resolution, cmap_light=ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])):
	x_max,y_max = max
	x_min, y_min = min

	# Draw boundry decision!.
	explt.set_xlim(x_min, x_max)
	explt.set_ylim(y_min, y_max)

	xx, yy, x1, x2 = mesh2Dgrid((x_min, y_min), (x_max, y_max), resolution)
	map = np.array([[_x, _y] for _x, _y in zip(x1, x2)])
	predict = model.predict(map)
	clz_mesh = predict.reshape(xx.shape)
	explt.pcolormesh(xx, yy, clz_mesh, cmap=cmap_light)
