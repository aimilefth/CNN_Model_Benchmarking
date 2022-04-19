import numpy as np
import tensorflow as tf

MODEL_PATH = "./Model_zip/LeNet5_Cifar10_47pct_0_5MF"
DATASET_PATH = "./cifar_test_1000.csv"
DATASET_LABELS_PATH = "./cifar_test_1000_labels.csv"
WARMUP=True

print("TF version:", tf.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

#Load the Dataset
x_test = np.loadtxt(DATASET_PATH, dtype=int, delimiter=",")
y_test = np.loadtxt(DATASET_LABELS_PATH, dtype=int, delimiter=",")
print(MODEL_PATH)
#Do the preprocessing that is necessary per model. 
if "ResNet50" in MODEL_PATH:
    x_test = x_test.reshape(-1,224,224,3)
    x_test = tf.keras.applications.resnet50.preprocess_input(x_test)
elif "NASNet_large" in MODEL_PATH:
    x_test = x_test.reshape(-1,224,224,3)
    x_test = tf.image.resize(x_test, (331, 331))
    x_test = tf.keras.applications.nasnet.preprocess_input(x_test)
elif "MobileNetV3" in MODEL_PATH:
    x_test = x_test.reshape(-1,224,224,3)
    x_test = tf.cast(x=x_test, dtype=tf.float32)/127.5 - 1.0
elif "LeNet5" in MODEL_PATH:
    x_test = x_test.reshape(-1,32,32,3)
    x_test = tf.image.rgb_to_grayscale(x_test)
    x_test = tf.cast(x=x_test, dtype=tf.float32)/255.0
# Load the model
model = tf.keras.models.load_model(filepath=MODEL_PATH)

print(model.layers[0].input_shape)
print(model.layers[0].dtype)

print(model.layers[-1].output_shape)
print(model.layers[-1].dtype)
# Experiments
from timeit import default_timer as timer
print(x_test.shape)
iterations = x_test.shape[0]

#Using call in for loop, passing 1 input
total_time=0.0
acc = 0
if(WARMUP == True):
    x_input = x_test[np.newaxis, 1, :]
    start = timer()
    output_data = model(x_input, training=False)
    end = timer()
    print("Time of first: %.3f ms" %((end - start)*1000))

for i in range(iterations):
    x_input = x_test[np.newaxis, i, :]
    start = timer()
    output_data = model(x_input, training=False)
    end = timer()
    #print("Time: %.3f ms" %((end - start)*1000))
    total_time += end - start # Time in seconds, e.g. 5.38091952400282
    if(np.argmax(output_data) == y_test[i]):
        acc += 1
print("Using call in for loop, passing 1 input")
avg_time = total_time/ iterations
acc = acc / iterations
print("Average time: %.3f ms" %(avg_time*1000))
print("Top1 Accuracy: %.4f" %acc)

#Using predict with batch size = 1
total_time = 0.0
acc = 0
start = timer()
output_data = model.predict(x=x_test, batch_size=1)
end = timer()
total_time += end - start # Time in seconds, e.g. 5.38091952400282
for i in range(iterations):
    if(np.argmax(output_data[i]) == y_test[i]):
        acc += 1
print("Using predict with batch size = 1")
avg_time = total_time/ iterations
acc = acc / iterations
print("Average time: %.3f ms" %(avg_time*1000))
print("Top1 Accuracy: %.4f" %acc)

#Using call in for loop with jit, passing 1 input
@tf.function(jit_compile=True)
def inference(model, x):
    return model(x, training=False)

total_time=0.0
acc = 0
if(WARMUP == True):
    x_input = x_test[np.newaxis, 1, :]
    start = timer()
    output_data = inference(model, x_input)
    end = timer()
    print("Time of first: %.3f ms" %((end - start)*1000))


    
for i in range(iterations):
    x_input = x_test[np.newaxis, i, :]
    start = timer()
    output_data = inference(model, x_input)
    end = timer()
    #print("Time: %.3f ms" %((end - start)*1000))
    total_time += end - start # Time in seconds, e.g. 5.38091952400282
    if(np.argmax(output_data) == y_test[i]):
        acc += 1
print("Using call in for loop with jit, passing 1 input")
avg_time = total_time/ iterations
acc = acc / iterations
print("Average time: %.3f ms" %(avg_time*1000))
print("Top1 Accuracy: %.4f" %acc)
