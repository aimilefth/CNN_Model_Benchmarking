import tensorflow as tf
import numpy as np
MODEL_NAME = "ResNet152_trick_35ep_41_9GF"
QUANTIZE = True
DATASET_PATH = "../ImageNet_val_100.csv"


def representative_data_gen():
  for input_value in tf.data.Dataset.from_tensor_slices(x_test).batch(1).take(50):
    # Model has only one input so each data point has one element.
    yield [input_value]


converter = tf.lite.TFLiteConverter.from_saved_model("./" + MODEL_NAME)
if(QUANTIZE):
   converter.optimizations = [tf.lite.Optimize.DEFAULT]
   x_test = np.loadtxt(DATASET_PATH, dtype=int, delimiter=",")
   if "ResNet50" in MODEL_NAME:
      x_test = x_test.reshape(-1,224,224,3)
      x_test = tf.keras.applications.resnet50.preprocess_input(x_test)
   elif "NASNet_large" in MODEL_NAME:
      x_test = x_test.reshape(-1,224,224,3)
      x_test = tf.image.resize(x_test, (331, 331))
      x_test = tf.keras.applications.nasnet.preprocess_input(x_test)
   elif "MobileNet" in MODEL_NAME:
      x_test = x_test.reshape(-1,224,224,3)
      x_test = tf.cast(x=x_test, dtype=tf.float32)/127.5 - 1.0
   elif ("ResNetV2152" in MODEL_NAME) or ("ResNet152" in MODEL_NAME) or ("InceptionV4" in MODEL_NAME):
      x_test = x_test.reshape(-1,224,224,3)
      x_test = tf.image.resize(x_test, (299, 299))
      x_test = tf.cast(x=x_test, dtype=tf.float32)/127.5 - 1.0
   elif "LeNet5" in MODEL_NAME:
      x_test = x_test.reshape(-1,32,32,3)
      x_test = tf.image.rgb_to_grayscale(x_test)
      x_test = tf.cast(x=x_test, dtype=tf.float32)/255.0
   converter.representative_dataset = representative_data_gen
   # Ensure that if any ops can't be quantized, the converter throws an error
   converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
   # Set the input and output tensors to uint8 (APIs added in r2.3)
   converter.inference_input_type = tf.uint8
   converter.inference_output_type = tf.uint8

tflite_model = converter.convert()

if(QUANTIZE):
   with open(MODEL_NAME+'_INT8.tflite', 'wb') as f:
      f.write(tflite_model)
else:
   with open(MODEL_NAME+'.tflite', 'wb') as f:
      f.write(tflite_model)
