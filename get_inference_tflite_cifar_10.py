import numpy as np
import tensorflow as tf

MODEL_NAME_TFLITE =
NUM_THREADS = None

#Load the Dataset
x_test = np.loadtxt("../cifar_test_1000.csv", dtype=int, delimiter=",")
y_test = np.loadtxt("../cifar_test_1000_labels.csv", dtype=int, delimiter=",")
x_test = x_test.reshape(-1,32,32,3)
#Do the preprocessing that is necessary per model. 
x_test = tf.image.rgb_to_grayscale(x_test)
x_test = tf.cast(x=x_test, dtype=tf.float32)/255.0
# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="./" + MODEL_NAME_TFLITE, num_threads=NUM_THREADS)
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
iterations = x_test.shape[0] #should be 1000
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
