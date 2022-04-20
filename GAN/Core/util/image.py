def generate_image(model, seed):
	return model(seed, training=False)
