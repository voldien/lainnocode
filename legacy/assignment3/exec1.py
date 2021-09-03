import os.path
import pickle

import numpy as np
import pandas as pd
from matplotlib import style
from matplotlib.colors import ListedColormap
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import ExtraTreeClassifier

style.use('fivethirtyeight')
np.set_printoptions(precision=4)
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])  # mesh plot
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])  # colors

picklePath = "exec1_model.sav"
pickleTransformPath = "exec1_transform.sav"


def processDataToNumerical(X, y):
	"""
	Process the data by converting each unique classifying value
	to an unique integer.
	:param X:
	:param y:
	:return:
	"""
	Xc = np.array(X)
	yc = np.array(y)

	# Process the explanatory variables.
	dirSet = []
	i = 0
	for x in X.T:
		dic = {}
		for _x in x:
			if isinstance(_x, str) and len(_x) > 0:
				if _x in dic.keys():
					dic[_x] = dic[_x]
				else:
					dic[_x] = i
					i = i + 1

		dirSet.append(dic)
	#    res = one_got.fit_transform(Xt)

	print(dirSet)
	for i, x in enumerate(Xc.T):
		if len(dirSet[i]) > 0:
			for j, _x in enumerate(x):
				if _x in dirSet[i].keys():
					x[j] = dirSet[i][_x]

	# Process the
	yDir = {}
	i = 0
	for _x in yc:
		o = _x[0]
		if isinstance(o, str):
			if o in yDir.keys():
				yDir[o] = yDir[o]
			else:
				yDir[o] = i
				i = i + 1

	for j, _x in enumerate(yc):
		o = _x[0]
		yc[j] = yDir[o]

	return Xc, yc, yDir


def exerciseTheModel(picklePath):
	k = [5, 10]
	parameters = {'max_depth': [2, 3, 5, 10, None], 'min_samples_leaf': [1, 2, 3, 4],
	              'criterion': ['gini', 'entropy'], }
	df = pd.read_csv("datasets/trainingDecisionTree.csv", header=None)

	X = np.array(df.drop(columns=41, axis=1))
	y = np.array(df.drop(columns=[i for i in range(0, 41)], axis=1))
	nf = len(X[0])
	variance = 0.01
	n_iter_search = 20

	prX, pry, yDir = processDataToNumerical(X, y)

	# Remove features that does little to no contribution on the data.
	print("Original datashape", prX.shape)
	var = VarianceThreshold(variance)
	prX = var.fit_transform(prX)
	# Save the model.
	with open(pickleTransformPath, 'wb') as f:
		pickle.dump(var, f)
	print("Shape with variance higher than {}".format(variance), prX.shape)

	yf = np.array([_y[0] for _y in pry])
	bestModel = [0, None]
	for _k in k:

		# Perform random grid search on forest tree.
		randomGrid = RandomizedSearchCV(estimator=ExtraTreeClassifier(), param_distributions=parameters, n_jobs=-1,
		                                cv=_k, n_iter=n_iter_search)
		randomGrid.fit(prX, y)

		# Perform random grid search on forest tree.
		grid = GridSearchCV(estimator=ExtraTreeClassifier(), param_grid=parameters, n_jobs=-1, cv=_k)
		grid.fit(prX, y)

		# Determine the best model.
		print("random Grid", randomGrid.best_params_)
		print("Grid Search ", grid.best_params_)
		if grid.best_score_ > randomGrid.best_score_:
			if bestModel[0] < grid.best_score_:
				bestModel[1] = grid.best_estimator_
				bestModel[0] = grid.best_score_
				model = grid.best_estimator_
		else:
			model = randomGrid.best_estimator_
			if randomGrid.best_score_ > bestModel[0]:
				bestModel[1] = randomGrid.best_estimator_
				bestModel[0] = randomGrid.best_score_

		print(grid.best_score_)
		print(grid.best_params_)
		print(model)
		print(grid.cv_results_)

	print("Final model:", bestModel[1], "score :", bestModel[0])
	# Save the model.
	with open(picklePath, 'wb') as f:
		pickle.dump(bestModel[1], f)


def testModel(model, transform, tx, ty):
	print(model)
	tx, _ty, diry = processDataToNumerical(tx, ty)

	print("Original datashape ", tx.shape)
	_tx = transform.transform(tx)
	print("Shape transformed", _tx.shape)
	score = model.score(_tx, ty)
	print(score)


if not os.path.exists(picklePath):
	exerciseTheModel(picklePath)
else:
	with open(picklePath, 'rb') as f:
		model = pickle.loads(f.read())
	with open(pickleTransformPath, 'rb') as f:
		transform = pickle.loads(f.read())

	#
	# df = pd.read_csv("datasets/trainingDecisionTree.csv", header=None)
	#
	# X = np.array(df.drop(columns=41, axis=1))
	# y = np.array(df.drop(columns=[i for i in range(0, 41)], axis=1))

	# TODO input data here!
	x_test = None
	y_test = None
	if x_test is None and y_test is None:
		raise ValueError("Test data not provided.")
	testModel(model, transform, x_test, y_test)
#	testModel(model, transform, X, y)
