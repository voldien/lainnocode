import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn import svm
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV


def plot_svc_decision_function(model, ax=None):
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()

    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    z = np.linspace(zlim[0], zlim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])

    # plot support vectors
    # ax.scatter(model.support_vectors_[:, 0],
    #            model.support_vectors_[:, 1],
    #            s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


filename = 'exec2_model.sav'
df = pd.read_csv("dataBM.csv", header=None)

X = np.array(df.drop(columns=2, axis=1))
y = np.array(df.drop(columns=[0, 1], axis=1)).ravel()

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

try:
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    ax = plt.axes(projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.scatter(X[:, 0], X[:, 1], model.predict(X), c=y, s=50, cmap='autumn')

    #ax.scatter(x_test[:, 0], x_test[:, 1], y_test, c=y, s=50, cmap='autumn')
    plot_svc_decision_function(model, ax)
    plt.suptitle("dataBM.csv score {}".format(model.score(x_test, y_test)))
    ax.view_init(elev=88., azim=-89.)

    plt.show()
except FileNotFoundError as f:
    k = [5, 10]
    # If model file does not exist. Create it.

    
    # The best model.
    # model = svm.SVC(C=1000000.0, cache_size=200, class_weight=None, coef0=0.0,
    #     decision_function_shape='ovr', degree=3, gamma=10, kernel='rbf',
    #     max_iter=-1, probability=False, random_state=None, shrinking=True,
    #     tol=0.001, verbose=False)

    for _k in k:
        parameters = {'C': [0.001, 0.1, 1, 10, 100, 10e5], 'gamma': [10, 1, 0.1, 0.01, 0.001],
                      'decision_function_shape': ['ovr', 'ovo'], 'kernel': ['rbf', 'linear', 'sigmoid']}

        grid = RandomizedSearchCV(estimator=svm.SVC(verbose=False), param_distributions=parameters, n_jobs=6, cv=_k,
                                  n_iter=20)
        grid.fit(x_train, y_train)
        print("k:", _k, "Score:", grid.score(x_test, y_test))
        print(grid.best_estimator_)
        model = grid.best_estimator_

    pickle.dump(model, open(filename, 'wb'))
