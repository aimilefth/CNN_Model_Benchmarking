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

from ctypes import *
from typing import List
import cv2
import numpy as np
import vart
import os
import pathlib
import xir
import threading
import time
import sys
import argparse
import shutil
import zipfile

MODEL = "ResNet50"
IMAGES_CSV = "ImageNet_val_100.csv" # The contained files in the zip
LABELS_CSV = "ImageNet_val_100_labels.csv"

divider = '------------------------------------'


#def preprocess_fn(image_path, fix_scale):
    #'''
    #Image pre-processing.
    #Rearranges from BGR to RGB then normalizes to range 0:1
    #and then scales by input quantization scaling factor
    #input arg: path of image file
    #return: numpy array
    #'''
    #image = cv2.imread(image_path)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #image = image * (1/255.0) * fix_scale
    #image = image.astype(np.int8)
    #return image

def tf_preprocess_input(x,mode):
    if not issubclass(x.dtype.type, np.floating):
        x = x.astype(np.float32, copy=False)
    if mode == 'tf':
        x /= 127.5
        x -= 1.
        return x
    elif mode == 'torch':
        x /= 255.
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
      # 'RGB'->'BGR'
        x = x[..., ::-1]
        mean = [103.939, 116.779, 123.68]
        std = None
  # Zero-center by mean pixel
    x[..., 0] -= mean[0]
    x[..., 1] -= mean[1]
    x[..., 2] -= mean[2]
    if std is not None:
        x[..., 0] /= std[0]
        x[..., 1] /= std[1]
        x[..., 2] /= std[2]
    return x

def preprocess_image(image, fix_scale):
    #Image normalization
    #Args:     Image and label
    #Returns:  normalized image and unchanged label
    
    if(MODEL == "ResNet50"):
        image = tf_preprocess_input(image, "caffe")
	#image = tf.keras.applications.resnet50.preprocess_input(image)
    elif(MODEL == "MobileNet"):
    	image = image.astype(np.float32, copy=False)/127.5 - 1.0
    elif(MODEL == "NASNet_large"):
        image = cv2.resize(image, dsize=(331,331), interpolation=cv2.INTER_CUBIC)
        image = tf_preprocess_input(image, "tf")
        #image = tf.keras.applications.nasnet.preprocess_input(image)
    elif(MODEL == "LeNet5"):
    	image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    	image = image.astype(np.float32, copy=False)/255.0
    image = image * fix_scale
    image = image.astype(np.int8)
    return image

def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
    assert graph is not None, "'graph' should not be None."
    root_subgraph = graph.get_root_subgraph()
    assert (root_subgraph is not None), "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0
    return [
        cs
        for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]


def runDPU(id,start,dpu,img):
    '''get tensor'''
    inputTensors = dpu.get_input_tensors()
    outputTensors = dpu.get_output_tensors()
    input_ndim = tuple(inputTensors[0].dims)
    output_ndim = tuple(outputTensors[0].dims)

    # we can avoid output scaling if use argmax instead of softmax
    #output_fixpos = outputTensors[0].get_attr("fix_point")
    #output_scale = 1 / (2**output_fixpos)

    batchSize = input_ndim[0]
    print("batchSize %d" %(batchSize))
    print("output_ndim %d" %(output_ndim[0]))
    n_of_images = len(img)
    print("n_of_images %d" %(n_of_images))
    count = 0
    write_index = start
    print("write_index %d" %(start))
    ids=[]
    ids_max = n_of_images+1
    outputData = []
    for i in range(ids_max):
        outputData.append([np.empty(output_ndim, dtype=np.int8, order="C")])
    while count < n_of_images:
        if (count+batchSize<=n_of_images):
            runSize = batchSize
        else:
            runSize=n_of_images-count

        '''prepare batch input/output '''
        inputData = []
        inputData = [np.empty(input_ndim, dtype=np.int8, order="C")]
        print("count, runSize %d %d" %(count, runSize))
        '''init input image to input buffer '''
        for j in range(runSize):
            imageRun = inputData[0]
            imageRun[j, ...] = img[(count + j) % n_of_images].reshape(input_ndim[1:])
        '''run with batch '''
        job_id = dpu.execute_async(inputData,outputData[len(ids)])
        ids.append((job_id,runSize,start+count))
        count = count + runSize 
        if count<n_of_images:
            if len(ids) < ids_max-1:
                continue
        for index in range(len(ids)):
            print(index)
            dpu.wait(ids[index][0]) # (1,0) tuple as input, strange
            write_index = ids[index][2] #
            '''store output vectors '''
            for j in range(ids[index][1]): #batchSize
                # we can avoid output scaling if use argmax instead of softmax
                # out_q[write_index] = np.argmax(outputData[0][j] * output_scale)
                out_q[write_index] = np.argmax(outputData[index][0][j])
                write_index += 1
        print(ids)
        ids=[]


def app(image_dir,csv_zip_dir,use_csv,threads,model):

    global out_q
    print("1")
    g = xir.Graph.deserialize(model)
    print("1")
    subgraphs = get_child_subgraph_dpu(g)
    print("1")
    all_dpu_runners = []
    print("1")
    for i in range(threads):
        all_dpu_runners.append(vart.Runner.create_runner(subgraphs[0], "run"))
    print("2")
    # input scaling
    input_fixpos = all_dpu_runners[0].get_input_tensors()[0].get_attr("fix_point")
    input_scale = 2**input_fixpos
    ''' preprocess images '''
    print (divider)
    if(use_csv):
        zip_ref = zipfile.ZipFile(csv_zip_dir, 'r')
        dataset_dir = "./temp"
        os.makedirs(dataset_dir, exist_ok = True) 
        zip_ref.extractall(dataset_dir)
        zip_ref.close()
        data = np.loadtxt(os.path.join(dataset_dir, IMAGES_CSV), dtype=int, delimiter=",")
        labels = np.loadtxt(os.path.join(dataset_dir, LABELS_CSV), dtype=int, delimiter=",")
        os.remove(os.path.join(dataset_dir, IMAGES_CSV))
        os.remove(os.path.join(dataset_dir, LABELS_CSV))
        if(MODEL == "LeNet5"):
            data = data.reshape(-1, 32, 32, 3)
        else:
            data = data.reshape(-1, 224, 224, 3)
    else: #WIP
        listimage=os.listdir(image_dir)
        runTotal = len(listimage)
    runTotal = data.shape[0]
    out_q = [None] * runTotal
    print('Pre-processing',runTotal,'images...')
    img = []
    for i in range(runTotal):
        if(use_csv):
            img.append(preprocess_image(data[i], input_scale))
        else:
            path = os.path.join(image_dir,listimage[i])
            img.append(preprocess_fn(path, input_scale))

    '''run threads '''
    print('Starting',threads,'threads...')
    threadAll = []
    start=0
    for i in range(threads):
        if (i==threads-1):
            end = len(img)
        else:
            end = start+(len(img)//threads)
        in_q = img[start:end]
        t1 = threading.Thread(target=runDPU, args=(i,start,all_dpu_runners[i], in_q))
        threadAll.append(t1)
        start=end

    time1 = time.time()
    for x in threadAll:
        x.start()
    for x in threadAll:
        x.join()
    time2 = time.time()
    timetotal = time2 - time1

    fps = float(runTotal / timetotal)
    print (divider)
    print("Throughput=%.2f fps, total frames = %.0f, time=%.4f seconds" %(fps, runTotal, timetotal))


    ''' post-processing '''
    correct = 0
    wrong = 0
    print('Post-processing',len(out_q),'images..')
    for i in range(len(out_q)):
        if(use_csv):
             if(out_q[i]==labels[i]):
                 correct+=1
             else:
                 wrong+=1
        else:
             prediction = classes[out_q[i]]
             ground_truth, _ = listimage[i].split('.',1)
             if (ground_truth==prediction):
                 correct += 1
             else:
                 wrong += 1
    accuracy = correct/len(out_q)
    print('Correct:%d, Wrong:%d, Accuracy:%.4f' %(correct,wrong,accuracy))
    print (divider)
    
    return



# only used if script is run as 'main' from command line
def main():

  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()  
  ap.add_argument('-d', '--image_dir', type=str, default='images', help='Path to folder of images. Default is images')
  ap.add_argument('-z', '--csv_zip_dir',type=str, default='./ImageNet_val_100.zip', help='Path to zip containing images in csv. Default is ImageNet_val_100.zip')
  ap.add_argument('-csv', '--use_csv',type=int, default=1, help='1 to use csv, 0 to not. Default is 1')    
  ap.add_argument('-t', '--threads',   type=int, default=1,        help='Number of threads. Default is 1')
  ap.add_argument('-m', '--model',     type=str, default='customcnn.xmodel', help='Path of xmodel. Default is customcnn.xmodel')

  args = ap.parse_args()  
  
  print ('Command line options:')
  print (' --image_dir : ', args.image_dir)
  print (' --csv_zip_dir:', args.csv_zip_dir)
  print (' --use_csv   : ', args.use_csv)
  print (' --threads   : ', args.threads)
  print (' --model     : ', args.model)

  app(args.image_dir,args.csv_zip_dir,args.use_csv,args.threads,args.model)

if __name__ == '__main__':
  main()

