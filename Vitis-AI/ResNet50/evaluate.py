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
Training script for dogs-vs-cats tutorial.
'''

'''
Author: Mark Harvey
'''

import os
import shutil
import sys
import argparse


# Silence TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import tensorflow as tf
from tensorflow import keras
import numpy as np

from dataset_utils import  input_fn_test_images, input_fn_test_csv

GLOBALS = '-----------------------------------------'
MODEL_NAME = "ResNet50_ImageNet_70_90_7_76GF"
VERBOSE = 1
ZIPFILE_NAME = 'ImageNet_val_100.zip'    # The zip file name inside files folder 
IMAGES_CSV = "ImageNet_val_100.csv" # The contained files in the zip
LABELS_CSV = "ImageNet_val_100_labels.csv"

DIVIDER = '-----------------------------------------'

def evaluate(data_dir,batchsize,chkpt_dir,use_csv):

    '''
    Define the model
    '''
    print("TF version:", tf.__version__)
    #model = tf.keras.models.load_model("./"+MODEL_NAME)
    if("ResNet50" in MODEL_NAME):
       model = tf.keras.applications.resnet50.ResNet50(
          include_top=True,
          weights='imagenet',
          input_tensor=None,
          input_shape=None,
          pooling=None,
          classes=1000,
          classifier_activation=None
       )
    elif("MobileNetV1" in MODEL_NAME):
       model = tf.keras.applications.MobileNet(
          include_top=True,
          weights='imagenet',
          input_tensor=None,
          input_shape=None,
          pooling=None,
          classes=1000,
          classifier_activation=None
       )
    model.compile(
       loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
       optimizer=keras.optimizers.Adam(),
       metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)]
    )
    print('\n'+DIVIDER)
    print(' Model Summary')
    print(DIVIDER)
    print(model.summary())
    print("Model Inputs: {ips}".format(ips=(model.inputs)))
    print("Model Outputs: {ops}".format(ops=(model.outputs)))


    '''
    tf.data pipelines
    '''
    print(DIVIDER)
    # train and test folders
    print("Testing...")
    if(use_csv):
    	test_dataset = input_fn_test_csv(ZIPFILE_NAME, IMAGES_CSV, LABELS_CSV, batchsize)
    else:
    	test_dataset = input_fn_test_images(data_dir, batchsize)
    model.evaluate(test_dataset, verbose = VERBOSE)
    print(DIVIDER)
    print(f"Saving the model to folder {chkpt_dir}")
    shutil.rmtree(chkpt_dir, ignore_errors=True) 
    os.makedirs(chkpt_dir, exist_ok = True)
    if(".h5" in MODEL_NAME):
    	shutil.copy(MODEL_NAME, os.path.join(chkpt_dir, MODEL_NAME)) 
    else: #Convert model to .h5
        model.save(os.path.join(chkpt_dir, "f_model.h5"))
    print(DIVIDER)
    return

def run_main():
    
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-cf', '--chkpt_dir',   type=str,   default='build/float_model', help='Path and name of folder for storing Keras checkpoints. Default is build/float_model')
    ap.add_argument('-d',  '--data_dir',   type=str,   default='build/dataset',    help='Path of folder containing Dataset Images. Default is build/dataset')
    ap.add_argument('-b',  '--batchsize',   type=int,   default=50,     help='Training batchsize. Must be an integer. Default is 50.')
    ap.add_argument('-csv','--use_csv', type=int, default=1, help="1 to use csv, 0 to not use csv. Default is 1.")
    args = ap.parse_args()

    print('\n'+DIVIDER)
    print('Keras version      : ',tf.keras.__version__)
    print('TensorFlow version : ',tf.__version__)
    print(sys.version)
    print(DIVIDER)
    print(' Command line options:')
    print ('--data_dir         : ',args.data_dir)
    print ('--batchsize        : ',args.batchsize)
    print ('--chkpt_dir        : ',args.chkpt_dir)
    print ('--use_csv          : ',args.use_csv)
    print(DIVIDER)
    
    evaluate(args.data_dir, args.batchsize,args.chkpt_dir,args.use_csv)


if __name__ == '__main__':
    run_main()
