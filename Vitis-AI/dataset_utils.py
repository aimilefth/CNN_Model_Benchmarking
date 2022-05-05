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
Utility functions for tf.data pipeline
'''

'''
Author: Mark Harvey, Xilinx Inc
'''
import os

# Silence TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import tensorflow as tf
import numpy as np
import pathlib
import shutil
import zipfile

MODEL = "ResNet50"

def normalize_image(image, label):
    #Image normalization
    #Args:     Image and label
    #Returns:  normalized image and unchanged label

    if(MODEL == "ResNet50"):
    	image = tf.keras.applications.resnet50.preprocess_input(image)
    elif(MODEL == "MobileNet"):
    	image = tf.cast(x=image, dtype=tf.float32)/127.5 - 1.0
    elif(MODEL == "NASNet_large"):
    	image = tf.image.resize(image, (331, 331))
    	image = tf.keras.applications.nasnet.preprocess_input(image)
    elif(MODEL == "LeNet5"):
    	image = tf.image.rgb_to_grayscale(image)
    	image = tf.cast(x=image, dtype=tf.float32)/255.0
    	
    return image, label

#Works for whole dataset
def input_fn_test_images(directory, batchsize):
	AUTOTUNE = tf.data.AUTOTUNE
	ds_val = tf.keras.utils.image_dataset_from_directory(
    		directory = directory,
    		labels = "inferred",
    		label_mode = "int",
    		color_mode = "rgb",
    		batch_size = batchsize,
    		image_size = (224, 224)
	)

	ds_val = ds_val.map(map_func=normalize_image, num_parallel_calls=AUTOTUNE)
	return ds_val

#Works for csv dataset 
def input_fn_test_csv(zip_directory, images_csv, labels_csv, batchsize):
	AUTOTUNE = tf.data.AUTOTUNE
	zip_ref = zipfile.ZipFile(zip_directory, 'r')
	dataset_dir = "./temp"
	os.makedirs(dataset_dir, exist_ok = True) 
	zip_ref.extractall(dataset_dir)
	zip_ref.close()
	data = np.loadtxt(os.path.join(dataset_dir, images_csv), dtype=int, delimiter=",")
	labels = np.loadtxt(os.path.join(dataset_dir, labels_csv), dtype=int, delimiter=",")
	os.remove(os.path.join(dataset_dir, images_csv))
	os.remove(os.path.join(dataset_dir, labels_csv))
	if(MODEL == "LeNet5"):
		data = data.reshape(-1, 32, 32, 3)
	else:
		data = data.reshape(-1, 224, 224, 3)
	ds_val = tf.data.Dataset.from_tensor_slices((data, labels))
	ds_val = ds_val.map(normalize_image, num_parallel_calls=AUTOTUNE)
	ds_val = ds_val.batch(batchsize)
	return ds_val
