import sys

from PIL import Image
import tensorflow as tf
import numpy as np
from skimage.color import lab2rgb, rgb2lab
import tensorflow_io as tfio




with tf.device('/device:CPU:0'):
	im = Image.open(sys.argv[1])
	im = im.convert('RGB')
	# px = im.load()
	encoder_model = tf.keras.models.load_model("stylized-encoder.h5")
	decoder_model = tf.keras.models.load_model("stylized-decoder.h5")
	# model = tf.keras.models.load_model("stylized-autoencoder.h5")

	encoder_model.summary()
	decoder_model.summary()
	input_width, input_height = (128, 128)  # encoder_model.input_shape[1:2]

	latent_values = []
	compressed_image = Image.new("RGB", im.size, im.getpixel((0, 0)))

	for x in range(0, int(im.width / input_width)):
		for y in range(0, int(im.height / input_height)):
			left = x * input_width
			top = y * input_height
			right = (x + 1) * input_width
			bottom = (y + 1) * input_height

			cropped_sub_image = im.crop((left, top, right, bottom))

			cropped_sub_image = rgb2lab((np.array(cropped_sub_image) * (1.0 / 255.0)).astype(dtype='float32')) * (1.0 / 128.0)
			cropped_sub_image = np.expand_dims(cropped_sub_image, axis=0)
			#cropped_sub_image = tf.convert_to_tensor(cropped_sub_image)
			#cropped_sub_image = tfio.experimental.color.rgb_to_lab(cropped_sub_image)

			latent = encoder_model(cropped_sub_image, training=False)

			decoder_image = decoder_model.predict(latent, verbose=0) * 128.0

			#decoder_image = model.predict(cropped_sub_image,verbose=0) * 128.0
			decoder_image = np.asarray(lab2rgb(decoder_image[0])).astype(dtype='float32')

			decoder_image_u8 = np.uint8(decoder_image * 255)
			compressed_crop_im = Image.fromarray(decoder_image_u8, "RGB")
			#compressed_crop_im.show()

			compressed_image.paste(compressed_crop_im, (left, top, right, bottom))

			latent_values.append(latent)
	# latent_values.
	full = np.concatenate(latent_values).flatten().astype(dtype='float16')
	np.savez_compressed('image_compressed', image=full)
	compressed_image.save("compressed.png")
