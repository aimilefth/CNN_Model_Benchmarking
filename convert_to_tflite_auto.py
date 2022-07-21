import tensorflow as tf
import numpy as np
import argparse
import logging
import os
from timeit import default_timer as timer
#Globals
DIVIDER = '-------------------------------------------------------------'



def representative_data_gen(x_test): #This might now work with argument in input
  for input_value in tf.data.Dataset.from_tensor_slices(x_test).batch(1).take(50):
    # Model has only one input so each data point has one element.
    yield [input_value]

def app(MODEL_NAME, QUANTIZE, DATASET_PATH):
   logging.info("Create Converter")
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
      converter.representative_dataset = representative_data_gen(x_test)
      # Ensure that if any ops can't be quantized, the converter throws an error
      converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
      # Set the input and output tensors to uint8 (APIs added in r2.3)
      converter.inference_input_type = tf.uint8
      converter.inference_output_type = tf.uint8
   logging.info("Start Conversion")
   tflite_model = converter.convert()
   logging.info("Save tflite model")
   if(QUANTIZE):
      with open(MODEL_NAME+'_INT8.tflite', 'wb') as f:
         f.write(tflite_model)
   else:
      with open(MODEL_NAME+'.tflite', 'wb') as f:
         f.write(tflite_model)

def main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-m', '--model_name',  type=str, default='ResNet50/ResNet50_ImageNet_70_90_7_76GF', help='Full path of model. Default is ResNet50/ResNet50_ImageNet_70_90_7_76GF')
    ap.add_argument('-q', '--quantize',  type=bool, default=False, help='Quantize to uint8. Default is false')
    ap.add_argument('-d', '--dataset_path',   type=str,   default=' ImageNet_val_100.csv', help='Path and name of folder for storing Keras checkpoints. Default is build/float_model')
    ap.add_argument('-l', '--log_file', type=str, default=None, help='Path and name of log file. Default is logs/convert_{model_name}.log')
    args = ap.parse_args()

    print(' Command line options:')
    print ('--chkpt_dir        : ',args.model_name)
    print ('--quantize         : ',str(args.quantize))
    print ('--dataset_path     : ',args.dataset_path)
    print ('--log_file         : ',args.log_file)
    
    print(DIVIDER)
    model_name = args.model_name.split("/")
    if(args.log_file == None):
        os.makedirs("logs", exist_ok=True)
        if(args.quantize):
            log_file = 'logs/convert_{}_INT8.log'.format(model_name[-1])
        else:
            log_file = 'logs/convert_{}.log'.format(model_name[-1])
        logging.basicConfig(filename=log_file, level=logging.INFO)
    else:
        logging.basicConfig(filename=args.log_file, level=logging.INFO)

    logging.info(' Command line options:')
    logging.info('--chkpt_dir        : {}'.format(args.model_name))
    logging.info('--quantize         : {}'.format(str(args.quantize))
    logging.info('--dataset_path     : {}'.format(args.dataset_path)
    logging.info('--log_file         : {}'.format(args.log_file)
    logging.info('{}'.format(DIVIDER))

    global_start_time = timer()
    app(args.model_name, args.quantize, args.dataset_path)
    global_end_time = timer()
    logging.info("Execution Time: %.3f", %(global_end_time - global_start_time))

if __name__ == '__main__':
    main()
