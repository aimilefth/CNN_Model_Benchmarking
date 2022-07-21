import numpy as np
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
import argparse
import logging
import os
import random
import torch
#import threading
from timeit import default_timer as timer
#Globals
DIVIDER = '-------------------------------------------------------------'


def preprocess(x_test, label, MODEL_PATH):
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

def my_input_fn(MODEL_PATH, BATCH_SIZE):
   if "ResNet50" in MODEL_PATH:
      in_shape = (1, BATCH_SIZE, 224, 224, 3)
   elif "NASNet_large" in MODEL_PATH:
      in_shape = (1, BATCH_SIZE,331, 331, 3)
   elif "MobileNet" in MODEL_PATH:
      in_shape = (1, BATCH_SIZE,224, 224, 3)
   elif "LeNet5" in MODEL_PATH:
      in_shape = (1, BATCH_SIZE, 32, 32 ,1)
   elif ("ResNetV2152" in MODEL_PATH) or ("ResNet152" in MODEL_PATH) or ("InceptionV4" in MODEL_PATH):
      in_shape = (1, BATCH_SIZE, 299, 299, 3)
   in1 = tf.zeros(shape=in_shape, dtype=tf.float32)
   yield in1

def calibration_input_fn(calibration_dataset):
   for x in calibration_dataset:
       yield (x[0], )

def app(MODEL_PATH, BATCH_SIZE, MODE, DATASET_PATH, DATASET_SIZE, PRINT_EVERY):
   logging.info("TF version: {}".format(tf.__version__))
   gpu_avail= "GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE"
   logging.info("{}".format(gpu_avail))
   OUTPUT_MODEL_PATH = MODEL_PATH + "_TensorRT_" + MODE + "_BATCH_" + str(BATCH_SIZE)
   logging.info("-------------Load_Data-------------")
   if("LeNet5" in MODEL_PATH):
      image_size = (32,32)
   else:
      image_size = (224,224)
   
   ds_val = tf.keras.preprocessing.image_dataset_from_directory(
      directory = DATASET_PATH,
      labels = "inferred",
      label_mode = "int",
      color_mode = "rgb",
      batch_size = BATCH_SIZE,
      image_size = image_size,
      shuffle = False
   )
   ds_val = ds_val.map(lambda x, y: preprocess(x, y, MODEL_PATH))
   logging.info("The TRT model exists?: {}".format(os.path.exists(OUTPUT_MODEL_PATH)))
   if not(os.path.exists(OUTPUT_MODEL_PATH)):
      logging.info("Convert the model with TF TRT")

# Usage of tf.experimental.tensorrt
#conversion_params = tf.experimental.tensorrt.ConversionParams(precision_mode="FP32")
#converter = tf.experimental.tensorrt.Converter(
#    input_saved_model_dir=MODEL_PATH,
#    conversion_params=conversion_params)

      if(MODE == "INT8"):
         logging.info("-------------Prepare Calibration Dataset-------------")
         num_calibration_batches = 10
         dataset = ds_val.shard(2,0)
         if(num_calibration_batches//((DATASET_SIZE / 2) // BATCH_SIZE) > 0):
            dataset = dataset.repeat(int(num_calibration_batches//((DATASET_SIZE / 2) // BATCH_SIZE))) # This guarantees that we have num_calibration_batches correctly
         calibration_dataset = dataset.take(num_calibration_batches)

      logging.info("-------------ConvParams--------------")
      conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
#conversion_params = tf.experimental.tensorrt.ConversionParams(
#    precision_mode=MODE,
#    use_calibration=False,
#    allow_build_at_runtime=False
#)
      conversion_params = conversion_params._replace(precision_mode=MODE)
      if(MODE == "INT8"):
         conversion_params = conversion_params._replace(use_calibration=True)
      logging.info("{}".format(conversion_params))
      logging.info("-------------Load--------------")
      converter = trt.TrtGraphConverterV2(
         input_saved_model_dir=MODEL_PATH,
         conversion_params=conversion_params)
#converter = tf.experimental.tensorrt.Converter(
#    input_saved_model_dir=MODEL_PATH, conversion_params=conversion_params) 
      logging.info("-------------Convert--------------")
      if(MODE == "INT8"):
         converter.convert(calibration_input_fn=lambda: calibration_input_fn(calibration_dataset))
      else:
         converter.convert()
      logging.info("-------------Build--------------")
      converter.build(input_fn=lambda: my_input_fn(MODEL_PATH, BATCH_SIZE))
      logging.info("-------------Save--------------")
      converter.save(OUTPUT_MODEL_PATH)
    
   logging.info("-------------LoadRT-------------")
   tensorRT_model_loaded = tf.saved_model.load(
	OUTPUT_MODEL_PATH, tags=[tag_constants.SERVING])
   graph_func = tensorRT_model_loaded.signatures[
	signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

   signature_keys = list(tensorRT_model_loaded.signatures.keys())
   logging.info("{}".format(signature_keys))

   logging.info("{}".format(graph_func.structured_outputs))

   logging.info("{}".format(graph_func))
   logging.info("-------------Experiment-------------")


   total_time=0.0
   acc = 0
   logging.info("Dataset size: {}".format(DATASET_SIZE))
   num_samples = DATASET_SIZE
   iterations = num_samples // BATCH_SIZE
   tested_samples = iterations * BATCH_SIZE
   WARMUP = True
   if(WARMUP):
      logging.info("Warmup")
      for element in ds_val.take(1):
         x_test = element[0]
      x_input = tf.constant(x_test[:])
      start = timer()
      output_data = graph_func(x_input)
      end = timer()
      #print(output_data[graph_func.structured_outputs['name']].shape)
      logging.info("First pass time %.3f ms" %((end - start)*1000))
   name = list(output_data.keys())[0]
   i = 0
   x = 0
   logging.info("Starting")
   for element in ds_val.take(iterations):
      x_test = element[0]
      y_test = element[1]
      x_input =tf.constant(x_test[:])
      #index = random.randrange(0, BATCH_SIZE) 
      start = timer()
      #output_data = graph_func(x_input)[name].numpy()
      output_data = graph_func(x_input) 
      torch.cuda.synchronize()
      #if(x < 1000.0): #This is actually 100% equal to if(true)
      #   end = timer()
      #Prints for sanity
      end = timer()
      #index2 = random.randrange(0, BATCH_SIZE)
      #start2 = timer()
      #y = output_data[name][index2,0]
      #end2 = timer()
      #print("{} {:.3f}".format(i, (end2-start2)*1000))
      if(i%PRINT_EVERY == 0):
         logging.info("{} {:.3f}".format(i, (end - start)*1000))
      #output_data = list(output_data.values())
      total_time += end - start # Time in seconds, e.g. 5.38091952400282
      #print(output_data.shape)
      for j in range(BATCH_SIZE):
         if(np.argmax(output_data[name][j]) == y_test[j]):
            acc += 1
      #y = output_data[name][index,0]
      #if(x != y):
      #    logging.info("{} {:.3f} {:.3f} shoot".format(i, x, y))
      i = i+BATCH_SIZE

   avg_time = total_time/iterations
   throughput = tested_samples/total_time
   acc = acc / tested_samples
   logging.info("Batch size %d, Mode %s, Iterations %d, Tested samples %d" %(BATCH_SIZE, MODE, iterations, tested_samples))
   logging.info("Average batch time: %.3f ms" %(avg_time*1000))
   logging.info("Throughput: %.3f fps" %(throughput))
   logging.info("Top1 Accuracy: %.4f" %acc)

   logging.info("Writing to results.txt")
   with open("results.txt", "a") as myfile:
      myfile.write("{} MODE: {} Batch: {:03d} Throughput:{:.3f}\n". format(MODEL_PATH, MODE, BATCH_SIZE, throughput))

def main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-m', '--model_path',   type=str, default='./Model_zip/ResNet50_ImageNet_70_90_7_76GF', help='Full path of model. Default is ./Model_zip/ResNet50_ImageNet_70_90_7_76GF')
    ap.add_argument('-b', '--batch_size',   type=int, default=1 , help='Batch Size. Default is 1')
    ap.add_argument('-t', '--data_type',    type=str, default='FP32', help="Data type for model. Default is FP32")
    ap.add_argument('-d', '--dataset_path', type=str, default='./ImageNet_val_1000', help='Path of dataset. Default is ./ImageNet_val_1000')
    ap.add_argument('-s', '--dataset_size', type=int, default=1000, help='Size of dataset or number of images to be checked. Default is 1000')
    ap.add_argument('-p', '--print_every',  type=int, default=64, help='How often sanity prints are made. Default is 64')
    ap.add_argument('-l', '--log_file',     type=str, default=None, help='Path and name of log file. Default is logs/convert_{model_name}_{data_type}_B{batch_size}_.log')
    args = ap.parse_args()

    print(' Command line options:')
    print ('--model_path       : ',args.model_path)
    print ('--batch_size       : ',str(args.batch_size))
    print ('--data_type        : ',args.data_type)
    print ('--dataset_path     : ',args.dataset_path)
    print ('--dataset_size     : ',str(args.dataset_size))
    print ('--print_every      : ',str(args.print_every))
    print ('--log_file         : ',args.log_file)
    
    print(DIVIDER)
    model_name = args.model_path.split("/")
    if(args.log_file == None):
        os.makedirs("logs", exist_ok=True)
        logging.basicConfig(filename='logs/GPU_inf_{}_{}_B{}.log'.format(model_name[-1], args.data_type, args.batch_size), level=logging.INFO)
    else:
        logging.basicConfig(filename=args.log_file, level=logging.INFO)

    logging.info(' Command line options:')
    logging.info('--model_name_tflite: {}'.format(args.model_path))
    logging.info('--batch_size       : {}'.format(str(args.batch_size)))
    logging.info('--data_type        : {}'.format(args.data_type))
    logging.info('--dataset_path     : {}'.format(args.dataset_path))
    logging.info('--dataset_size     : {}'.format(str(args.dataset_size)))
    logging.info('--print_every      : {}'.format(str(args.print_every)))
    logging.info('--log_file         : {}'.format(args.log_file))
    logging.info('{}'.format(DIVIDER))

    global_start_time = timer()
    app(args.model_path, args.batch_size, args.data_type, args.dataset_path, args.dataset_size, args.print_every)
    global_end_time = timer()
    logging.info("Execution Time: %.3f" %(global_end_time - global_start_time))

if __name__ == '__main__':
    main()
