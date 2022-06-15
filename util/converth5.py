import sys

import tensorflow as tf
import numpy as np
from tensorflow import keras, lite


keras_file = sys.argv[1]
print(keras_file)
model = tf.keras.models.load_model(keras_file)

converter = lite.TFLiteConverter.from_keras_model(model)
#converter.allow_custom_ops = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.experimental_new_converter = True
converter.target_spec.supported_types = [tf.float32]
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32
converter.post_training_quantize = True
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]

#converter.representative_dataset = representative_dataset
tfmodel = converter.convert()

open(sys.argv[1] + ".tflite", "wb").write(tfmodel)
