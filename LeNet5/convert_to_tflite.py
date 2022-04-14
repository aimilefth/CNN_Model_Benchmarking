import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model("./LeNet5_Cifar10_47pct_0_5MF")
tflite_model = converter.convert()

with open('LeNet5_Cifar10_47pct_0_5MF.tflite', 'wb') as f:
	f.write(tflite_model)
