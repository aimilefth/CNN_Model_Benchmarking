import tensorflow as tf
MODEL_NAME = 
converter = tf.lite.TFLiteConverter.from_saved_model("./" + MODEL_NAME)
tflite_model = converter.convert()

with open(MODEL_NAME+'.tflite', 'wb') as f:
	f.write(tflite_model)
