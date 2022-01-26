import numpy as np
import tensorflow.keras.datasets.fashion_mnist
import tensorflow.keras.datasets.cifar100


def loadDataFashion():
	(train_images, train_labels), (test_images, test_labels) = tensorflow.keras.datasets.fashion_mnist.load_data()
	class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
	               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

	train_images = np.expand_dims(train_images, axis=-1)  # <--- add batch axis
	test_images = np.expand_dims(test_images, axis=-1)  # <--- add batch axis
	return np.concatenate((train_images, test_images)), np.concatenate((train_labels, test_labels)), class_names

def loadDataCifar10():
	(train_images, train_labels), (test_images, test_labels) = tensorflow.keras.datasets.cifar10.load_data()
	class_names =["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "ship", "horse", "ship", "truck"]
	return np.concatenate((train_images, test_images)), np.concatenate((train_labels, test_labels)), class_names

def loadDataCifar100():
	(train_images, train_labels), (test_images, test_labels) = tensorflow.keras.datasets.cifar100.load_data()
	class_names = [
		"apple",
		"aquarium_fish",
		"baby",
		"bear",
		"beaver",
		"bed",
		"bee",
		"beetle",
		"bicycle",
		"bottle",
		"bowl",
		"boy",
		"bridge",
		"bus",
		"butterfly",
		"camel",
		"can",
		"castle",
		"caterpillar",
		"cattle",
		"chair",
		"chimpanzee",
		"clock",
		"cloud",
		"cockroach",
		"couch",
		"crab",
		"crocodile",
		"cup",
		"dinosaur",
		"dolphin",
		"elephant",
		"flatfish",
		"forest",
		"fox",
		"girl",
		"hamster",
		"house",
		"kangaroo",
		"keyboard",
		"lamp",
		"lawn_mower",
		"leopard",
		"lion",
		"lizard",
		"lobster",
		"man",
		"maple_tree",
		"motorcycle",
		"mountain",
		"mouse",
		"mushroom",
		"oak_tree",
		"orange",
		"orchid",
		"otter",
		"palm_tree",
		"pear",
		"pickup_truck",
		"pine_tree",
		"plain",
		"plate",
		"poppy",
		"porcupine",
		"possum",
		"rabbit",
		"raccoon",
		"ray",
		"road",
		"rocket",
		"rose",
		"sea",
		"seal",
		"shark",
		"shrew",
		"skunk",
		"skyscraper",
		"snail",
		"snake",
		"spider",
		"squirrel",
		"streetcar",
		"sunflower",
		"sweet_pepper",
		"table",
		"tank",
		"telephone",
		"television",
		"tiger",
		"tractor",
		"train",
		"trout",
		"tulip",
		"turtle",
		"wardrobe",
		"whale",
		"willow_tree",
		"wolf",
		"woman",
		"worm"]

	return np.concatenate((train_images, test_images)), np.concatenate((train_labels, test_labels)), class_names
