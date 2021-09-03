import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier

modelPath = "exec4_model.sav"


def drawNeuronNetwork(ax, nr_layers, layers, biases):
    """
    https://gist.github.com/dvgodoy/0db802cfb8edd488dfbd524302ca4be7
    Small alteration of making it works in python3.7
    :param ax:
    :param nr_layers:
    :param layers:
    :param biases:
    :return:
    """
    left, right, bottom, top = (.1, .9, .1, .9)
    print("weights:", layers)
    print("bias:", biases)

    coefs_ = biases
    intercepts_ = layers

    ax.axis('off')

    layer_sizes = [2, 2, 1]
    n_layers = len(layer_sizes)
    v_spacing = (top - bottom) / float(max(layer_sizes))
    h_spacing = (right - left) / float(len(layer_sizes) - 1)

    # Input-Arrows
    layer_top_0 = v_spacing * (layer_sizes[0] - 1) / 2. + (top + bottom) / 2.
    for m in range(layer_sizes[0]):
        plt.arrow(left - 0.18, layer_top_0 - m * v_spacing, 0.12, 0, lw=1, head_width=0.01, head_length=0.02)

    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing * (layer_size - 1) / 2. + (top + bottom) / 2.
        for m in range(layer_size):
            circle = plt.Circle((n * h_spacing + left, layer_top - m * v_spacing), v_spacing / 8.,
                                color='w', ec='k', zorder=4)
            if n == 0:
                plt.text(left - 0.125, layer_top - m * v_spacing, r'$X_{' + str(m + 1) + '}$', fontsize=15)
            elif (n_layers == 3) & (n == 1):
                plt.text(n * h_spacing + left + 0.00, layer_top - m * v_spacing + (v_spacing / 8. + 0.01 * v_spacing),
                         r'$H_{' + str(m + 1) + '}$', fontsize=15)
            elif n == n_layers - 1:
                plt.text(n * h_spacing + left + 0.10, layer_top - m * v_spacing, r'$y_{' + str(m + 1) + '}$',
                         fontsize=15)
            ax.add_artist(circle)
    # Bias-Nodes
    for n, layer_size in enumerate(layer_sizes):
        if n < n_layers - 1:
            x_bias = (n + 0.5) * h_spacing + left
            y_bias = top + 0.005
            circle = plt.Circle((x_bias, y_bias), v_spacing / 8., color='w', ec='k', zorder=4)
            plt.text(x_bias - (v_spacing / 8. + 0.10 * v_spacing + 0.01), y_bias, r'$1$', fontsize=15)
            ax.add_artist(circle)
            # Edges
    # Edges between nodes
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing * (layer_size_a - 1) / 2. + (top + bottom) / 2.
        layer_top_b = v_spacing * (layer_size_b - 1) / 2. + (top + bottom) / 2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n * h_spacing + left, (n + 1) * h_spacing + left],
                                  [layer_top_a - m * v_spacing, layer_top_b - o * v_spacing], c='k')
                ax.add_artist(line)
                xm = (n * h_spacing + left)
                xo = ((n + 1) * h_spacing + left)
                ym = (layer_top_a - m * v_spacing)
                yo = (layer_top_b - o * v_spacing)
                rot_mo_rad = np.arctan((yo - ym) / (xo - xm))
                rot_mo_deg = rot_mo_rad * 180. / np.pi
                xm1 = xm + (v_spacing / 8. + 0.05) * np.cos(rot_mo_rad)
                if n == 0:
                    if yo > ym:
                        ym1 = ym + (v_spacing / 8. + 0.12) * np.sin(rot_mo_rad)
                    else:
                        ym1 = ym + (v_spacing / 8. + 0.05) * np.sin(rot_mo_rad)
                else:
                    if yo > ym:
                        ym1 = ym + (v_spacing / 8. + 0.12) * np.sin(rot_mo_rad)
                    else:
                        ym1 = ym + (v_spacing / 8. + 0.04) * np.sin(rot_mo_rad)
                plt.text(xm1, ym1, \
                         str(round(coefs_[n][m, o], 4)), \
                         rotation=rot_mo_deg, \
                         fontsize=10)
    # Edges between bias and nodes
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        if n < n_layers - 1:
            layer_top_a = v_spacing * (layer_size_a - 1) / 2. + (top + bottom) / 2.
            layer_top_b = v_spacing * (layer_size_b - 1) / 2. + (top + bottom) / 2.
        x_bias = (n + 0.5) * h_spacing + left
        y_bias = top + 0.005
        for o in range(layer_size_b):
            line = plt.Line2D([x_bias, (n + 1) * h_spacing + left],
                              [y_bias, layer_top_b - o * v_spacing], c='k')
            ax.add_artist(line)
            xo = ((n + 1) * h_spacing + left)
            yo = (layer_top_b - o * v_spacing)
            rot_bo_rad = np.arctan((yo - y_bias) / (xo - x_bias))
            rot_bo_deg = rot_bo_rad * 180. / np.pi
            xo2 = xo - (v_spacing / 8. + 0.01) * np.cos(rot_bo_rad)
            yo2 = yo - (v_spacing / 8. + 0.01) * np.sin(rot_bo_rad)
            xo1 = xo2 - 0.05 * np.cos(rot_bo_rad)
            yo1 = yo2 - 0.05 * np.sin(rot_bo_rad)
            plt.text(xo1, yo1, \
                     str(round(intercepts_[n][o], 4)), \
                     rotation=rot_bo_deg, \
                     fontsize=10)

            # Output-Arrows
    layer_top_0 = v_spacing * (layer_sizes[-1] - 1) / 2. + (top + bottom) / 2.
    for m in range(layer_sizes[-1]):
        plt.arrow(right + 0.015, layer_top_0 - m * v_spacing, 0.16 * h_spacing, 0, lw=1, head_width=0.01,
                  head_length=0.02)
    # # Record the n_iter_ and loss
    # plt.text(left + (right - left) / 3., bottom - 0.005 * v_spacing, \
    #          'Steps:' + str(n_iter_) + '    Loss: ' + str(round(loss_, 6)), fontsize=15)

    plt.title("Neuron network")


def assertDataSet(X, y):
    for _x, _y in zip(X, y):
        x1, x2 = _x
        if x1 == x2:
            assert _y == 0
        elif x1 != x2:
            assert _y == 1
        else:
            assert False


def generateDataSet(nSamples=100000):
    # X = np.array([[random.randint(0, 1), random.randint(0, 1)] for _ in range(nSamples)])
    template = [[0, 0], [1, 0], [1, 1], [0, 1]]
    nFSamples = int(nSamples / 4)
    X_ = [_x for _ in range(0, nFSamples) for _x in template]
    X = np.array(X_)
    assert len(X) == nSamples
    y = np.array([x[0] ^ x[1] for x in X]).reshape(len(X), 1)

    assertDataSet(X, y)

    hotY = []
    for _y in y:
        if _y == 0:
            hotY.append((1, 0))
        else:
            hotY.append((0, 1))
    y = np.array(hotY)

    return X, y


def trainModel(savePath):
    X, y = generateDataSet()

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

    parameters = {'activation': ['logistic'], 'alpha': [1e-5, 1e-4, 1e-1], 'solver': ['sgd', 'adam'],
                  'hidden_layer_sizes': [(2,)]}

    clf = MLPClassifier(max_iter=300, verbose=True, )
    grid = GridSearchCV(estimator=clf, param_grid=parameters, cv=10, n_jobs=-1)
    grid.fit(x_train, y_train)
    model = grid.best_estimator_
    print("Best model score", model.score(x_test, y_test))
    print(model)
    print("weights:", model.coefs_)
    print("bias:", model.intercepts_)

    # Save the model.
    with open(savePath, 'wb') as f:
        pickle.dump(model, f)
    return model


try:
    with open(modelPath, 'rb') as f:
        model = pickle.loads(f.read())
except FileNotFoundError as f:
    model = trainModel(modelPath)

# Draw the model.
fig = plt.figure(figsize=(12, 12))

drawNeuronNetwork(fig.gca(), model.n_layers_, np.array(model.intercepts_), np.array(model.coefs_))
plt.show()
fig.savefig('exec4_network.png')
