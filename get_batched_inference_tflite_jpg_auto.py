import numpy as np
import tensorflow as tf
import argparse
import logging
import os
import random
from timeit import default_timer as timer
#Globals
DIVIDER = '-------------------------------------------------------------'

def preprocess(x_test, label, MODEL_NAME_TFLITE, input_details_0):
   if "ResNet50" in MODEL_NAME_TFLITE:
      #x_test = x_test.reshape(-1,224,224,3)
      x_test = tf.keras.applications.resnet50.preprocess_input(x_test)
   elif "NASNet_large" in MODEL_NAME_TFLITE:
      #x_test = x_test.reshape(-1,224,224,3)
      x_test = tf.image.resize(x_test, (331, 331))
      x_test = tf.keras.applications.nasnet.preprocess_input(x_test)
   elif "MobileNet" in MODEL_NAME_TFLITE:
      #x_test = x_test.reshape(-1,224,224,3)
      x_test = tf.cast(x=x_test, dtype=tf.float32)/127.5 - 1.0
   elif "LeNet5" in MODEL_NAME_TFLITE:
      #x_test = x_test.reshape(-1,32,32,3)
      x_test = tf.image.rgb_to_grayscale(x_test)
      x_test = tf.cast(x=x_test, dtype=tf.float32)/255.0
   elif ("ResNetV2152" in MODEL_NAME_TFLITE) or ("ResNet152" in MODEL_NAME_TFLITE) or ("InceptionV4" in MODEL_NAME_TFLITE):
      #x_test = x_test.reshape(-1,224,224,3)
      x_test = tf.image.resize(x_test, (299, 299))
      x_test = tf.cast(x=x_test, dtype=tf.float32)/127.5 - 1.0
   # If models are int8
   if(input_details_0['dtype'] == np.uint8):
      input_scale, input_zero_point = input_details_0["quantization"]
      x_test = x_test / input_scale + input_zero_point
      x_test = tf.cast(x=x_test, dtype=tf.uint8)
   return x_test, label

def app(MODEL_NAME_TFLITE, NUM_THREADS, BATCH_SIZE, DATASET_PATH, DATASET_SIZE, PRINT_EVERY):
   logging.info("Load the TFLite model and allocate tensors.")
   interpreter = tf.lite.Interpreter(model_path="./" + MODEL_NAME_TFLITE, num_threads=NUM_THREADS)
   #interpreter.allocate_tensors()

   input_details = interpreter.get_input_details()
   #output_details = interpreter.get_output_details()
   interpreter.resize_tensor_input(input_details[0]['index'], [BATCH_SIZE, input_details[0]['shape'][1], input_details[0]['shape'][2], input_details[0]['shape'][3]])
   interpreter.allocate_tensors()
   input_details = interpreter.get_input_details()
   output_details = interpreter.get_output_details()

   logging.info("{}".format(input_details[0]['shape']))
   logging.info("{}".format(input_details[0]['dtype']))
   logging.info("{}".format(output_details[0]['shape']))
   logging.info("{}".format(output_details[0]['dtype']))
   logging.info("Load the Dataset")

   if("LeNet5" in MODEL_NAME_TFLITE):
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

   ds_val = ds_val.map(lambda x, y: preprocess(x, y, MODEL_NAME_TFLITE, input_details[0]))
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
      start = timer()
      interpreter.set_tensor(input_details[0]['index'], x_test)
      interpreter.invoke()
      output_data = interpreter.get_tensor(output_details[0]['index'])
      end = timer()
      logging.info("First pass time %.3f ms" %((end - start)*1000))
   i = 0
   logging.info("Starting")
   for element in ds_val.take(iterations):
      x_test = element[0]
      y_test = element[1]
      index = random.randrange(0, BATCH_SIZE)
      start = timer()
      interpreter.set_tensor(input_details[0]['index'], x_test)
      interpreter.invoke()
      output_data = interpreter.get_tensor(output_details[0]['index'])
      end = timer()
      output_data[index,0] = output_data[index,0] + 1.0
      end2 = timer()
      output_data[index,0] = output_data[index,0] - 1.0
      logging.info("{} {:.3f}".format(i, (end2 - end)*1000))
      total_time += end - start # Time in seconds, e.g. 5.38091952400282
      #Prints for sanity
      if(i%PRINT_EVERY == 0):
         logging.info("{} {:.3f}".format(i, (end - start)*1000))
      for j in range(BATCH_SIZE):
         if(np.argmax(output_data[j,:]) == y_test[j]):
            acc += 1
      i = i+BATCH_SIZE

   avg_time = total_time/iterations
   throughput = tested_samples/total_time
   acc = acc / tested_samples
   logging.info("Batch size %d, Threads %d, Iterations %d, Tested samples %d" %(BATCH_SIZE, NUM_THREADS, iterations, tested_samples))
   logging.info("Average batch time: %.3f ms" %(avg_time*1000))
   logging.info("Throughput: %.3f fps" %(throughput))
   logging.info("Top1 Accuracy: %.4f" %acc)

   logging.info("Writing to results.txt")
   with open("results.txt", "a") as myfile:
      myfile.write("{} Threads: {:02d} Batch: {:03d} Throughput:{:.3f}\n". format(MODEL_NAME_TFLITE, NUM_THREADS, BATCH_SIZE, throughput))

def main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-m', '--model_name_tflite',  type=str, default='ResNet50/ResNet50_ImageNet_70_90_7_76GF.tflite', help='Full path of model. Default is ResNet50/ResNet50_ImageNet_70_90_7_76GF.tflite')
    ap.add_argument('-t', '--num_threads',  type=int, default=None, help='Number of threads to be used. Default is None')
    ap.add_argument('-b', '--batch_size',   type=int, default=1 , help='Batch Size. Default is 1')
    ap.add_argument('-d', '--dataset_path', type=str, default='ImageNet_val_1000', help='Path of dataset. Default is ImageNet_val_1000')
    ap.add_argument('-s', '--dataset_size', type=int, default=1000, help='Size of dataset or number of images to be checked. Default is 1000')
    ap.add_argument('-p', '--print_every',  type=int, default=64, help='How often sanity prints are made. Default is 64')
    ap.add_argument('-l', '--log_file',     type=str, default=None, help='Path and name of log file. Default is logs/convert_{model_name}_T{num_threads}_B{batch_size}.log')
    args = ap.parse_args()

    print(' Command line options:')
    print ('--model_name_tflite: ',args.model_name_tflite)
    print ('--num_threads      : ',str(args.num_threads))
    print ('--batch_size       : ',str(args.batch_size))
    print ('--dataset_path     : ',args.dataset_path)
    print ('--dataset_size     : ',str(args.dataset_size))
    print ('--print_every      : ',str(args.print_every))
    print ('--log_file         : ',args.log_file)
    
    print(DIVIDER)
    model_name = args.model_name_tflite.split("/")
    if(args.log_file == None):
        os.makedirs("logs", exist_ok=True)
        logging.basicConfig(filename='logs/inf_{}_T{}_B{}.log'.format(model_name[-1], args.num_threads, args.batch_size), level=logging.INFO)
    else:
        logging.basicConfig(filename=args.log_file, level=logging.INFO)

    logging.info(' Command line options:')
    logging.info('--model_name_tflite: {}'.format(args.model_name_tflite))
    logging.info('--num_threads      : {}'.format(str(args.num_threads)))
    logging.info('--batch_size       : {}'.format(str(args.batch_size)))
    logging.info('--dataset_path     : {}'.format(args.dataset_path))
    logging.info('--dataset_size     : {}'.format(str(args.dataset_size)))
    logging.info('--print_every      : {}'.format(str(args.print_every)))
    logging.info('--log_file         : {}'.format(args.log_file))
    logging.info('{}'.format(DIVIDER))

    global_start_time = timer()
    app(args.model_name_tflite, args.num_threads, args.batch_size, args.dataset_path, args.dataset_size, args.print_every)
    global_end_time = timer()
    logging.info("Execution Time: %.3f" %(global_end_time - global_start_time))

if __name__ == '__main__':
    main()
