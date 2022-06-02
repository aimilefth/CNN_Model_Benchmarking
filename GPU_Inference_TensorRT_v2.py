import numpy as np
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants

print("TF version:", tf.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

MODEL_PATH = "./Model_zip/MobileNetV1_ImageNet_69_87_1_15GF"
DATASET_PATH = "./ImageNet_val_100"
MODE = "FP32"
OUTPUT_MODEL_PATH = MODEL_PATH + "_TensorRT_" + MODE
BATCH_SIZE = 8

def preprocess(x_test, label):
   if "ResNet50" in MODEL_PATH:
      #x_test = x_test.reshape(-1,224,224,3)
      x_test = tf.keras.applications.resnet50.preprocess_input(x_test)
   elif "NASNet_large" in MODEL_PATH:
      #x_test = x_test.reshape(-1,224,224,3)
      x_test = tf.image.resize(x_test, (331, 331))
      x_test = tf.keras.applications.nasnet.preprocess_input(x_test)
   elif "MobileNet" in MODEL_PATH:
      #x_test = x_test.reshape(-1,224,224,3)
      x_test = tf.cast(x=x_test, dtype=tf.float32)/127.5 - 1.0
   elif "LeNet5" in MODEL_PATH:
      #x_test = x_test.reshape(-1,32,32,3)
      x_test = tf.image.rgb_to_grayscale(x_test)
      x_test = tf.cast(x=x_test, dtype=tf.float32)/255.0
   elif ("ResNetV2152" in MODEL_PATH) or ("ResNet152" in MODEL_PATH) or ("InceptionV4" in MODEL_PATH):
      #x_test = x_test.reshape(-1,224,224,3)
      x_test = tf.image.resize(x_test, (299, 299))
      x_test = tf.cast(x=x_test, dtype=tf.float32)/127.5 - 1.0
   return x_test, label

print("-------------Load_Data-------------")
#Load the Dataset

ds_val = tf.keras.preprocessing.image_dataset_from_directory(
    directory = DATASET_PATH,
    labels = "inferred",
    label_mode = "int",
    color_mode = "rgb",
    batch_size = 1,
    image_size = (224, 224),
    shuffle = False
)

ds_val = ds_val.map(preprocess)

# Load the model
#model = tf.keras.models.load_model(filepath=MODEL_PATH)

# Convert the model with TensorflowRT

def my_input_fn():
	if "ResNet50" in MODEL_PATH:
		in_shape = (1, 1, 224, 224, 3)
	elif "NASNet_large" in MODEL_PATH:
		in_shape = (1, 1,331, 331, 3)
	elif "MobileNet" in MODEL_PATH:
		in_shape = (1, 1,224, 224, 3)
	elif "LeNet5" in MODEL_PATH:
		in_shape = (1, 1, 32, 32 ,1)
	elif ("ResNetV2152" in MODEL_PATH) or ("ResNet152" in MODEL_PATH) or ("InceptionV4" in MODEL_PATH):
		in_shape = (1, 1, 299, 299, 3)
	in1 = tf.zeros(shape=in_shape, dtype=tf.float32)
	yield in1
# Usage of tf.experimental.tensorrt
#conversion_params = tf.experimental.tensorrt.ConversionParams(precision_mode="FP32")
#converter = tf.experimental.tensorrt.Converter(
#    input_saved_model_dir=MODEL_PATH,
#    conversion_params=conversion_params)

if(MODE == "INT8"):
   print("-------------Prepare Calibration Dataset-------------")
   num_calibration_batches = 10
   dataset = tf.data.Dataset.from_tensor_slices(x_test)
   dataset = dataset.batch(batch_size=BATCH_SIZE)
   dataset = dataset.repeat(None)
   calibration_dataset = dataset.take(num_calibration_batches)
   def calibration_input_fn():
      for x in calibration_dataset:
          yield (x, )

print("-------------ConvParams--------------")
conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
#conversion_params = tf.experimental.tensorrt.ConversionParams(
#    precision_mode=MODE,
#    use_calibration=False,
#    allow_build_at_runtime=False
#)
conversion_params = conversion_params._replace(precision_mode=MODE)
if(MODE == "INT8"):
   conversion_params = conversion_params._replace(use_calibration=True)
print(conversion_params)
print("-------------Load--------------")
converter = trt.TrtGraphConverterV2(
    input_saved_model_dir=MODEL_PATH,
    conversion_params=conversion_params)
#converter = tf.experimental.tensorrt.Converter(
#    input_saved_model_dir=MODEL_PATH, conversion_params=conversion_params) 
print("-------------Convert--------------")
if(MODE == "INT8"):
   converter.convert(calibration_input_fn=calibration_input_fn)
else:
   converter.convert()
print("-------------Build--------------")
converter.build(input_fn=my_input_fn)
print("-------------Save--------------")
converter.save(OUTPUT_MODEL_PATH)

print("-------------LoadRT-------------")
tensorRT_model_loaded = tf.saved_model.load(
	OUTPUT_MODEL_PATH, tags=[tag_constants.SERVING])
graph_func = tensorRT_model_loaded.signatures[
	signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

signature_keys = list(tensorRT_model_loaded.signatures.keys())
print(signature_keys)

print(graph_func.structured_outputs)

print(graph_func)
print("-------------Experiment-------------")
from timeit import default_timer as timer
#print(x_test.shape)
iterations = 100  #x_test.shape[0]
for element in ds_val.take(1):
   x_test = element[0]
print(x_test)
print(x_test.shape)
total_time=0.0
acc = 0
WARMUP = 1
for i in range(WARMUP):
    x_input = tf.constant(x_test[:])
    start = timer()
    output_data = graph_func(x_input)
    end = timer()
    print("Time of first input: %.3f ms" %((end - start)*1000))
for i in range(iterations):
    for element in ds_val.take(1):
       x_test = element[0]
    x_input =tf.constant(x_test[:])
    start = timer()
    output_data = graph_func(x_input)
    end = timer()
    output_data = list(output_data.values())
    total_time += end - start # Time in seconds, e.g. 5.38091952400282
    if(np.argmax(output_data) == element[1]):
       acc += 1
print("Transferring %d inputs" %iterations)
avg_time = total_time/ iterations
acc = acc / iterations
print("Average time: %.3f ms" %(avg_time*1000))
print("Top1 Accuracy: %.4f" %acc)
