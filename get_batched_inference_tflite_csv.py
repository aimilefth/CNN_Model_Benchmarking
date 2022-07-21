import numpy as np
import tensorflow as tf

MODEL_NAME_TFLITE = "MobileNetV1/MobileNetV1_ImageNet_69_87_1_15GF.tflite"
NUM_THREADS = 1
BATCH_SIZE = 1
DATASET_PATH = "ImageNet_val_100.csv"
DATASET_LABELS_PATH = "ImageNet_val_100_labels.csv"
#Load the Dataset
x_test = np.loadtxt(DATASET_PATH, dtype=int, delimiter=",")
y_test = np.loadtxt(DATASET_LABELS_PATH, dtype=int, delimiter=",")
if("ImageNet" in DATASET_PATH):
    x_test = x_test.reshape(-1,224,224,3)
elif("cifar" in DATASET_PATH):
    x_test = x_test.reshape(-1,32,32,3)
#Do the preprocessing that is necessary per model. 
if "ResNet50" in MODEL_NAME_TFLITE:
	x_test = tf.keras.applications.resnet50.preprocess_input(x_test)
elif "NASNet_large" in MODEL_NAME_TFLITE:
	x_test = tf.image.resize(x_test, (331, 331))
	x_test = tf.keras.applications.nasnet.preprocess_input(x_test)
elif "MobileNet" in MODEL_NAME_TFLITE:
	x_test = tf.cast(x=x_test, dtype=tf.float32)/127.5 - 1.0
elif ("ResNetV2152" in MODEL_NAME_TFLITE) or ("ResNet152" in MODEL_NAME_TFLITE) or ("InceptionV4" in MODEL_NAME_TFLITE):
        x_test = tf.image.resize(x_test, (299, 299))
        x_test = tf.cast(x=x_test, dtype=tf.float32)/127.5 - 1.0
elif "LeNet5" in MODEL_NAME_TFLITE:
        x_test = tf.image.rgb_to_grayscale(x_test)
        x_test = tf.cast(x=x_test, dtype=tf.float32)/255.0
# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="./" + MODEL_NAME_TFLITE, num_threads=NUM_THREADS)
#interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
#output_details = interpreter.get_output_details()
interpreter.resize_tensor_input(input_details[0]['index'], [BATCH_SIZE, input_details[0]['shape'][1], input_details[0]['shape'][2], input_details[0]['shape'][3]])
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(input_details[0]['shape'])
print(input_details[0]['dtype'])

print(output_details[0]['shape'])
print(output_details[0]['dtype'])

if input_details[0]['dtype'] == np.uint8:
      input_scale, input_zero_point = input_details[0]["quantization"]
      x_test = x_test / input_scale + input_zero_point
      x_test = tf.cast(x=x_test, dtype=tf.uint8)

from timeit import default_timer as timer
total_time=0.0
acc = 0
print(x_test.shape)
num_samples= x_test.shape[0] #should be 100
iterations = num_samples // BATCH_SIZE
tested_samples = iterations * BATCH_SIZE
print(iterations, BATCH_SIZE, tested_samples)
WARMUP = True
#WARMUP
if(WARMUP):
    start = timer()
    if(BATCH_SIZE == 1):
        interpreter.set_tensor(input_details[0]['index'], x_test[np.newaxis, 0,:])
    else:
        interpreter.set_tensor(input_details[0]['index'], x_test[0:BATCH_SIZE,:])
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    end = timer()
    print("First pass time %.3f ms" %((end - start)*1000))

for i in range(iterations):
    start = timer()
    x = i*BATCH_SIZE
    if(BATCH_SIZE == 1):
    	interpreter.set_tensor(input_details[0]['index'], x_test[np.newaxis, i,:])
    else:
        interpreter.set_tensor(input_details[0]['index'], x_test[x:(x+BATCH_SIZE),:])
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    end = timer()
    total_time += end - start # Time in seconds, e.g. 5.38091952400282
    print((end - start)*1000)
    for j in range(BATCH_SIZE):
        if(np.argmax(output_data[j,:]) == y_test[x + j]):
            acc += 1
avg_time = total_time/iterations
throughput = tested_samples/total_time
acc = acc / tested_samples
print("Average batch time: %.3f ms" %(avg_time*1000))
print("Throughput: %.3f fps" %(throughput))
print("Top1 Accuracy: %.4f" %acc)
