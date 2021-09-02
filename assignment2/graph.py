import numpy as np


def meshgrid(X1, X2, h):
    """

    :param X1:
    :param X2:
    :param h:
    :return:
    """
    # Max and min of the bounding world.
    x_min, x_max = X1.min() - 0.1,  X1.max() + 0.1
    y_min, y_max = X2.min() - 0.1, X2.max() + 0.1

    #
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))  # Mesh Grid

    x1, x2 = xx.ravel(), yy.ravel()  # Turn to two Nx1 arrays
    return xx, yy, x1, x2


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