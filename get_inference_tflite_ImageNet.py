import numpy as np
import tensorflow as tf

MODEL_NAME_TFLITE =

#Load the Dataset
x_test = np.loadtxt("../ImageNet_val_100.csv", dtype=int, delimiter=",")
y_test = np.loadtxt("../ImageNet_val_100_labels.csv", dtype=int, delimiter=",")
x_test = x_test.reshape(-1,224,224,3)
#Do the preprocessing that is necessary per model. 
if "ResNet50" in MODEL_NAME_TFLITE:
	x_test = tf.keras.applications.resnet50.preprocess_input(x_test)
elif "NASNet_large" in MODEL_NAME_TFLITE:
	x_test = tf.image.resize(x_test, (331, 331))
	x_test = tf.keras.applications.nasnet.preprocess_input(x_test)
elif "MobileNetV3" in MODEL_NAME_TFLITE:
	x_test = tf.cast(x=x_test, dtype=tf.float32)/127.5 - 1.0

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="./" + MODEL_NAME_TFLITE)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(input_details[0]['shape'])
print(input_details[0]['dtype'])

print(output_details[0]['shape'])
print(output_details[0]['dtype'])

from timeit import default_timer as timer
total_time=0.0
acc = 0
print(x_test.shape)
iterations = x_test.shape[0] #should be 100
for i in range(iterations):
    start = timer()
    interpreter.set_tensor(input_details[0]['index'], x_test[np.newaxis, i,:])
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    end = timer()
    total_time += end - start # Time in seconds, e.g. 5.38091952400282
    if(np.argmax(output_data) == y_test[i]):
        acc += 1
avg_time = total_time/ iterations
acc = acc / iterations
print("Average time: %.3f ms" %(avg_time*1000))
print("Top1 Accuracy: %.4f" %acc)
