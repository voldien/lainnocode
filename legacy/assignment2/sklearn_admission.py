import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression

# Read CSV
X1, X2, y = np.genfromtxt('admission.csv', delimiter=",", unpack=True)

# Degree 2
Xe = np.c_[X1, X2, X1 ** 2, X1 * X2, X2 ** 2]  # No 1-column added!

logreg = LogisticRegression(solver='lbfgs', C=100, tol=1e-5)  # instantiate the model
logreg.fit(Xe, y)  # fit the model with datasets

y_pred = logreg.predict(Xe)  # predict training set
errors = np.sum(y_pred != y)
print('Training errors: ', errors)

# Assign a class to each point in the mesh [x_min, x_max]x[y_min, y_max].

# Setup mesh grid
h = .05  # step size in the mesh
x_min, x_max = X1.min() - 1, X1.max() + 1
y_min, y_max = X2.min() - 1, X2.max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))  # Mesh Grid
x1, x2 = xx.ravel(), yy.ravel()  # Two Nx1 vectors

#  predict each mesh point
xy_mesh = np.c_[x1, x2, x1 ** 2, x1 * x2, x2 ** 2]
classes = logreg.predict(xy_mesh)
clz_mesh = classes.reshape(xx.shape)

# Create mesh plot
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
plt.figure(1)
plt.pcolormesh(xx, yy, clz_mesh, cmap=cmap_light)
plt.scatter(X1, X2, c=y, marker='.', cmap=cmap_bold)
plt.title('admission.csv, using sklearn LogisticRegression')
plt.show()
