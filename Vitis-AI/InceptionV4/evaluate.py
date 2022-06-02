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
import Inception

GLOBALS = '-----------------------------------------'
MODEL_NAME = "InceptionV4_24_6GF"
VERBOSE = 1
ZIPFILE_NAME = 'ImageNet_val_100.zip'    # The zip file name inside files folder 
IMAGES_CSV = "ImageNet_val_100.csv" # The contained files in the zip
LABELS_CSV = "ImageNet_val_100_labels.csv"

DIVIDER = '-----------------------------------------'

#This is a function to add/remove layers from a given model
import re
from tensorflow.keras.models import Model

def insert_layer_nonseq(model, layer_regex, insert_layer_factory,
                        insert_layer_name=None, position='after'):

    # Auxiliary dictionary to describe the network graph
    network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}

    # Set the input layers of each layer
    for layer in model.layers:
        for node in layer._outbound_nodes:
            layer_name = node.outbound_layer.name
            if layer_name not in network_dict['input_layers_of']:
                network_dict['input_layers_of'].update(
                        {layer_name: [layer.name]})
            else:
                network_dict['input_layers_of'][layer_name].append(layer.name)

    # Set the output tensor of the input layer
    network_dict['new_output_tensor_of'].update(
            {model.layers[0].name: model.input})

    # Iterate over all layers after the input
    model_outputs = []
    for layer in model.layers[1:]:

        # Determine input tensors
        layer_input = [network_dict['new_output_tensor_of'][layer_aux] 
                for layer_aux in network_dict['input_layers_of'][layer.name]]
        if len(layer_input) == 1:
            layer_input = layer_input[0]

        # Insert layer if name matches the regular expression
        if re.match(layer_regex, layer.name):
            if position == 'replace':
                x = layer_input
            elif position == 'after':
                x = layer(layer_input)
            elif position == 'before':
                pass
            else:
                raise ValueError('position must be: before, after or replace')

            new_layer = insert_layer_factory()
            x = new_layer(x)
            print('New layer: {} Old layer: {} Type: {}'.format(new_layer.name,
                                                            layer.name, position))
            if position == 'before':
                x = layer(x)
        else:
            x = layer(layer_input)

        # Set new output tensor (the original one, or the one of the inserted
        # layer)
        network_dict['new_output_tensor_of'].update({layer.name: x})

        # Save tensor in output list if it is output in initial model
    if layer_name in model.output_names:
        model_outputs.append(x)
    return Model(inputs=model.inputs, outputs=model_outputs, name="resnet152v2_relu")

def Relu_1_layer_factory():
    return tf.keras.layers.Activation(keras.activations.relu, name="My_Relu_1")
def Relu_2_layer_factory():
    return tf.keras.layers.Activation(keras.activations.relu, name="My_Relu_2")
def Relu_3_layer_factory():
    return tf.keras.layers.Activation(keras.activations.relu, name="My_Relu_3")
def BN_layer_factory():
    return tf.keras.layers.BatchNormalization(name="My_BN")

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
    elif("ResNet152" in MODEL_NAME):
       model = tf.keras.applications.resnet.ResNet152(
          include_top=True,
          weights=None,
          input_tensor=None,
          input_shape=(299,299,3),
          pooling=None,
          classes=1000,
          classifier_activation=None
          )
    elif("InceptionV4" in MODEL_NAME):
       model = Inception.inception_v4((299,299,3), 1000, True)
       '''
       #This adds a Relu layer between first Conv2D and ZeroPadding2D to try to work around a certain Vitis-AI quantizer bug
       #model_full = insert_layer_nonseq(model=model_full, layer_regex='conv1_conv', insert_layer_factory=BN_layer_factory,
                       position='after')
       model_full.save("./temp")
       model_full = tf.keras.models.load_model("./temp")
       model_full = insert_layer_nonseq(model=model_full, layer_regex='My_BN', insert_layer_factory=Relu_1_layer_factory,
                 position='after')
       model_full.save("./temp")
       model_full = tf.keras.models.load_model("./temp")
       model_full = insert_layer_nonseq(model=model_full, layer_regex='conv2_block1_preact_bn', insert_layer_factory=Relu_2_layer_factory,
                 position='replace')
       model_full.save("./temp")
       model_full = tf.keras.models.load_model("./temp")
       model_full = insert_layer_nonseq(model=model_full, layer_regex='conv1_pad', insert_layer_factory=Relu_3_layer_factory,
                 position='replace')
       model_full.summary()
       model_input = keras.Input(shape=(299,299,3), name="input")
       features = model_full(model_input)
       output = keras.layers.Dense(units=1000, activation=None)(features)
       model = keras.Model(model_input, output, name="ResNet_V2_152_299")
       '''
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
