import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import style
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split

from regression import linear
from statistics import corr, SE

style.use('fivethirtyeight')
np.set_printoptions(precision=4)

df = pd.read_csv("datasets/data_build_stories.csv", header=None)

# Train Data.
X = np.array(df.drop(columns=1, axis=1))
y = np.array(df.drop(columns=0, axis=1))

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
# gradient(Xe, beta, )
model = LinearRegression()
model.fit(x_train, y_train)
predict = model.predict(x_test)
plt.scatter(x_train, y_train)
plt.scatter(x_test, predict)
plt.suptitle("Error {}".format(model.score(x_test, y_test)))
plt.show()

print("Coefficient {}".format(model.coef_.tolist()))
co = corr(X, y)
print("correlation: ", co)

# model.coef_[0] = 0
lineX = []
lineY = []
ei = []
lineX.append(0)
lineX.append(X.max())
lineY.append(0 * model.coef_[0])
lineY.append(X.max() * model.coef_[0])
for _x, _y in zip(X, y):
    fy = _x * model.coef_[0]
    ei.append((fy - _y) ** 2)

nErr = len(ei)
MSE = nErr / sum(ei)
plt.plot(lineX, lineY, c='r')
plt.scatter(x_train, y_train, c='b')
plt.scatter(x_test, predict, c='g')
plt.suptitle("MSE {}".format(MSE))
plt.show()


b1 = model.coef_[0]
beta = [0, b1]
predict = [linear(beta, _x) for _x in X]

# Compute the confident
CI = [b1 - 2.0 * SE(X, y, predict), b1 + 2.0 * SE(X, y, predict)]
print("CI (Confident interval)", CI)
