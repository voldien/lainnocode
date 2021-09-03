import os
import pickle

import PIL.Image as Image
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageFilter
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def load_face_dataset(resize=(96, 96), path="./faces"):
    cacheFileX = "cacheX.sav"
    cacheFileY = "cacheY.sav"

    try:
        with open(cacheFileX, 'rb') as f:
            X = pickle.loads(f.read())
        with open(cacheFileY, 'rb') as f:
            y = pickle.loads(f.read())
    except FileNotFoundError as f:
        # Load data
        X = []
        y = []
        for characterName in os.listdir(path):
            charNamePath = "{}/{}".format(path, characterName)
            if os.path.isdir(charNamePath):
                for imagePaths in os.listdir(charNamePath):
                    finalImagePath = "{}/{}".format(charNamePath, imagePaths)
                    img = Image.open(finalImagePath).convert('L').filter(ImageFilter.FIND_EDGES)
                    NPX = np.array(img)
                    X.append(NPX.reshape(96 * 96))
                    y.append(characterName)

        with open(cacheFileX, 'wb') as f:
            pickle.dump(X, f)
        with open(cacheFileY, 'wb') as f:
            pickle.dump(y, f)

    return X, y


X, y = load_face_dataset()  # load_dataset(NrChar=4,nSamples=1000)

# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(y)
print(integer_encoded)
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)
yhot = onehot_encoded

# Process the features values.
scaler = preprocessing.StandardScaler()
nX = scaler.fit_transform(X)
print("scalar mean", scaler.mean_)

x_train, x_test, y_train, y_test = train_test_split(nX, yhot, test_size=0.2, shuffle=True)
parameters = {'activation': ['logistic', 'relu'], 'alpha': [1e-5, 1e-4, 1e-2, 1e-1], 'solver': ['sgd'],
              'hidden_layer_sizes': [(10, 30, 20), (100, 50, 10), (20, 30), (15, 10), (10, 15), (10, 10)]}

plt.figure(figsize=(40, 5))
nSamples = 20
for index, (image, label) in enumerate(zip(x_train[0:nSamples], y_train[0:nSamples])):
    plt.subplot(nSamples / 5, 5, index + 1)
    plt.imshow(np.reshape(image, (96, 96)), cmap=plt.cm.gray)
    hotName = onehot_encoder.inverse_transform([label])

    nameIndex = int(hotName[0])
    name = label_encoder.inverse_transform([nameIndex])[0]
    plt.title('{}'.format(name), fontsize=20)
plt.show()

clf = MLPClassifier()
grid = GridSearchCV(clf, param_grid=parameters, cv=10, n_jobs=7)
grid.fit(x_train, y_train)
print(grid.score(x_test, y_test))
bestModel = grid.best_estimator_
print(bestModel)

# Save the model.
modelPath = "exec5_model.sav"
with open(modelPath, 'wb') as f:
    pickle.dump(bestModel, f)
