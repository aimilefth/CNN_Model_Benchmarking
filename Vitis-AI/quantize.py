'''
 Copyright 2020 Xilinx Inc.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
'''

'''
Quantize the floating-point model
'''

'''
Author: Mark Harvey
'''


import argparse
import os
import shutil
import sys

# Silence TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import tensorflow as tf
from tensorflow_model_optimization.quantization.keras import vitis_quantize
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from dataset_utils import input_fn_test_images, input_fn_test_csv

DIVIDER = '-----------------------------------------'

ZIPFILE_NAME = 'ImageNet_val_100.zip'    # The zip file name inside files folder 
IMAGES_CSV = "ImageNet_val_100.csv" # The contained files in the zip
LABELS_CSV = "ImageNet_val_100_labels.csv"

def quant_model(float_model,quant_model,batchsize,data_dir,evaluate,use_csv):
    '''
    Quantize the floating-point model
    Save to HDF5 file
    '''

    # make folder for saving quantized model
    head_tail = os.path.split(quant_model)
    os.makedirs(head_tail[0], exist_ok = True)

    # load the floating point trained model
    float_model = load_model(float_model)

    # get input dimensions of the floating-point model
    height = float_model.input_shape[1]
    width = float_model.input_shape[2]
    channels = float_model.input_shape[3]
    print(height)
    print(width)
    print(channels)
    if(use_csv):
    	quant_dataset = input_fn_test_csv(ZIPFILE_NAME, IMAGES_CSV, LABELS_CSV, batchsize)
    else:
    	quant_dataset = input_fn_test_images(data_dir, batchsize)
    print(DIVIDER)
    print(DIVIDER)
    # run quantization
    quantizer = vitis_quantize.VitisQuantizer(float_model)
    quantized_model = quantizer.quantize_model(calib_dataset=quant_dataset)

    # saved quantized model
    quantized_model.save(quant_model)
    print('Saved quantized model to',quant_model)


    if (evaluate):
        '''
        Evaluate quantized model
        '''
        print('\n'+DIVIDER)
        print ('Evaluating quantized model..')
        print(DIVIDER+'\n')
        if(use_csv):
        	test_dataset = input_fn_test_csv(ZIPFILE_NAME, IMAGES_CSV, LABELS_CSV, batchsize)
        else:
        	test_dataset = input_fn_test_images(data_dir, batchsize)
        quantized_model.compile(optimizer=Adam(),
                                loss='sparse_categorical_crossentropy', #We need to know this in order to make accurate evaluation
                                metrics=['accuracy'])

        scores = quantized_model.evaluate(test_dataset,
                                          verbose=1)
	
        print('Quantized model accuracy: {0:.4f}'.format(scores[1]*100),'%')
        quantized_model.summary()
        print('\n'+DIVIDER)

    return



def main():

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-m', '--float_model',  type=str, default='build/float_model/f_model.h5', help='Full path of floating-point model. Default is build/float_model/f_model.h5')
    ap.add_argument('-q', '--quant_model',  type=str, default='build/quant_model/q_model.h5', help='Full path of quantized model. Default is build/quant_model/q_model.h5')
    ap.add_argument('-b', '--batchsize',    type=int, default=50,                       help='Batchsize for quantization. Default is 50')
    ap.add_argument('-d', '--data_dir',type=str, default='build/dataset',              help='Path of folder containing Dataset Images. Default is build/dataset')
    ap.add_argument('-e', '--evaluate',     action='store_true', help='Evaluate floating-point model if set. Default is evaluation.')
    ap.add_argument('-csv','--use_csv', type=int, default=1, help="1 to use csv, 0 to not use csv. Default is 1.")
    args = ap.parse_args()

    print('\n------------------------------------')
    print('TensorFlow version : ',tf.__version__)
    print(sys.version)
    print('------------------------------------')
    print ('Command line options:')
    print (' --float_model  : ', args.float_model)
    print (' --quant_model  : ', args.quant_model)
    print (' --batchsize    : ', args.batchsize)
    print (' --data_dir     : ', args.data_dir)
    print (' --evaluate     : ', args.evaluate)
    print (' --use_csv      : ', args.use_csv)
    print('------------------------------------\n')


    quant_model(args.float_model, args.quant_model, args.batchsize, args.data_dir, args.evaluate, args.use_csv)


if __name__ ==  "__main__":
    main()
