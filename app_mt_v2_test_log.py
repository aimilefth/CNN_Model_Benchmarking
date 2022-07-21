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
import logging
from timeit import default_timer as timer

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

def preprocess_image(image, fix_scale, MODEL):
    #Image normalization
    #Args:     Image and label
    #Returns:  normalized image and unchanged label
    
    if(MODEL == "ResNet50"):
        image = tf_preprocess_input(image, "caffe")
	#image = tf.keras.applications.resnet50.preprocess_input(image)
    elif(MODEL == "MobileNetV1"):
    	image = image.astype(np.float32, copy=False)/127.5 - 1.0
    elif(MODEL == "LeNet5"):
        image = image.astype(np.float32, copy=False)/255.0
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = image.reshape(32, 32, 1)
    	#image = image.astype(np.float32, copy=False)/255.0
    elif(MODEL == "ResNet152" or MODEL == "InceptionV4"):
        image = image.astype(np.float32, copy=False)/127.5 - 1.0
        image = cv2.resize(image, dsize=(299,299), interpolation=cv2.INTER_CUBIC)
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


def runDPU(id,start,dpu,img,my_time,batch):
    '''get tensor'''
    inputTensors = dpu.get_input_tensors()
    outputTensors = dpu.get_output_tensors()
    input_ndim = tuple(inputTensors[0].dims)
    output_ndim = tuple(outputTensors[0].dims)

    # we can avoid output scaling if use argmax instead of softmax
    #output_fixpos = outputTensors[0].get_attr("fix_point")
    #output_scale = 1 / (2**output_fixpos)
    logging.info("input_ndim: {}".format(input_ndim))
    logging.info("output_ndim: {}".format(output_ndim))
    #batchSize = input_ndim[0]
    if(batch == 0):
	    batchSize = input_ndim[0]
    elif(batch > input_ndim[0]):
	    batchSize = input_ndim[0]
	    logging.info("Batch {} was bigger than max {}. Resized to {}".format(batch, input_ndim[0], input_ndim[0]))
    elif(batch < 0):
	    batchSize = input_ndim[0]
	    logging.info("Invalid size for batch {}. Resized to {}".format(batch, input_ndim[0]))
    elif(batch > 0 and batch <= input_ndim[0]):
	    batchSize = batch
    else:
	    print("Unexpected Error")
    logging.info("batchSize %d" %(batchSize))
    logging.info("output_ndim %d" %(output_ndim[0]))
    n_of_images = len(img)
    logging.info("n_of_images %d" %(n_of_images))
    count = 0
    write_index = start
    logging.info("write_index %d" %(start))
    outputData = []
    time_total = 0.0
    for i in range(n_of_images):
        outputData.append([np.empty(output_ndim, dtype=np.int8, order="C")])
    while count < n_of_images:
        if (count+batchSize<=n_of_images):
            runSize = batchSize
        else:
            runSize=n_of_images-count

        '''prepare batch input/output '''
        inputData = []
        inputData = [np.empty(input_ndim, dtype=np.int8, order="C")]
        time1 = time.time()
        #inputData[0] = img[count:count+runSize]
        imageRun = inputData[0]
        imageRun[0:runSize] = img[count:count+runSize]
        #inputData = [np.empty(input_ndim, dtype=np.int8, order="C")]
        #print("count, runSize %d %d" %(count, runSize))
        #'''init input image to input buffer '''
        #for j in range(runSize):
        #    imageRun = inputData[0]
        #    imageRun[j, ...] = img[(count + j) % n_of_images].reshape(input_ndim[1:])
        '''run with batch '''
        job_id = dpu.execute_async(inputData,outputData[count])
        dpu.wait(job_id)
        time2 = time.time()
        for j in range(runSize):
            out_q[start+count+j] = np.argmax(outputData[count][0][j])
        time_total += time2 - time1
        #ids.append((job_id,runSize,start+count))
        count = count + runSize 
        #if count<n_of_images:
        #    if len(ids) < ids_max-1:
        #        continue
        #for index in range(len(ids)):
        #    print(index)
        #    dpu.wait(ids[index][0]) # (1,0) tuple as input, strange
        #    write_index = ids[index][2] #
        #    '''store output vectors '''
        #    for j in range(ids[index][1]): #batchSize
                # we can avoid output scaling if use argmax instead of softmax
                # out_q[write_index] = np.argmax(outputData[0][j] * output_scale)
        #        out_q[write_index] = np.argmax(outputData[index][0][j])
        #        write_index += 1
        #print(ids)
        #ids=[]
    my_time[id] = time_total
    logging.info("id: {} time: {:.3f}".format(id, time_total))


def app(image_dir,csv_zip_dir,use_csv,threads,model,batch,model_name):

    global out_q
    logging.info("1")
    g = xir.Graph.deserialize(model)
    logging.info("Deserialized")
    subgraphs = get_child_subgraph_dpu(g)
    logging.info("Got Subgraphs")
    all_dpu_runners = []
    logging.info("Creating {} Runners".format(threads))
    for i in range(threads):
        all_dpu_runners.append(vart.Runner.create_runner(subgraphs[0], "run"))
    print("Created {} Runners".format(threads))
    logging.info("Created {} Runners".format(threads))
    # input scaling
    input_fixpos = all_dpu_runners[0].get_input_tensors()[0].get_attr("fix_point")
    input_scale = 2**input_fixpos
    ''' preprocess images '''
    print (divider)
    if(use_csv):
        zip_ref = zipfile.ZipFile(csv_zip_dir, 'r')
        name = csv_zip_dir.split('.')
        #print(name)
        IMAGES_CSV = name[0] + ".csv" # The contained files in the zip
        LABELS_CSV = name[0] + "_labels.csv"
        dataset_dir = "./temp"
        os.makedirs(dataset_dir, exist_ok = True) 
        zip_ref.extractall(dataset_dir)
        zip_ref.close()
        data = np.loadtxt(os.path.join(dataset_dir, IMAGES_CSV), dtype=int, delimiter=",")
        labels = np.loadtxt(os.path.join(dataset_dir, LABELS_CSV), dtype=int, delimiter=",")
        os.remove(os.path.join(dataset_dir, IMAGES_CSV))
        os.remove(os.path.join(dataset_dir, LABELS_CSV))
        if(model_name == "LeNet5"):
            data = data.reshape(-1, 32, 32, 3)
        else:
            data = data.reshape(-1, 224, 224, 3)
    else: #WIP
        listimage=os.listdir(image_dir)
        runTotal = len(listimage)
    runTotal = data.shape[0]
    out_q = [None] * runTotal
    logging.info('Pre-processing {} images...'.format(runTotal))
    img = []
    for i in range(runTotal):
        if(use_csv):
            img.append(preprocess_image(data[i], input_scale, model_name))
        else:
            path = os.path.join(image_dir,listimage[i])
            img.append(preprocess_fn(path, input_scale))

    '''run threads '''
    logging.info('Starting {} threads,'.format(threads))
    threadAll = []
    start=0
    my_time = [None] * threads
    for i in range(threads):
        if (i==threads-1):
            end = len(img)
        else:
            end = start+(len(img)//threads)
        in_q = img[start:end]
        t1 = threading.Thread(target=runDPU, args=(i,start,all_dpu_runners[i], in_q, my_time, batch))
        threadAll.append(t1)
        start=end

    time1 = time.time()
    for x in threadAll:
        x.start()
    for x in threadAll:
        x.join()
    time2 = time.time()
    timetotal = time2 - time1
    timetotal_v2 =  max(my_time)
    logging.info("timetotal %.3f , time_total_v2 %.3f" %(timetotal, timetotal_v2)) 
    #fps = float(runTotal / timetotal_v2)
    fps = float(runTotal/timetotal)
    logging.info("{}".format(divider))
    #logging.info("Throughput=%.2f fps, total frames = %d, time=%.3f seconds" %(fps, runTotal, timetotal_v2))
    logging.info("Throughput=%.2f fps, total frames = %d, time=%.3f seconds" %(fps, runTotal, timetotal))


    ''' post-processing '''
    correct = 0
    wrong = 0
    logging.info('Post-processing {} images..'.format(len(out_q)))
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
    logging.info('Correct:%d, Wrong:%d, Accuracy:%.4f' %(correct,wrong,accuracy))
    logging.info("{}".format(divider))
    
    logging.info("Writing to results.txt")
    with open("results.txt", "a") as myfile:
      myfile.write("{} Threads: {:02d} Batch: {:03d} Throughput:{:.3f}\n". format(model_name, threads, batch, fps))
      
    return



# only used if script is run as 'main' from command line
def main():

  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()  
  ap.add_argument('-d', '--image_dir', type=str, default='images', help='Path to folder of images. Default is images')
  ap.add_argument('-z', '--csv_zip_dir',type=str, default='ImageNet_val_100.zip', help='Path to zip containing images in csv. Default is ImageNet_val_100.zip')
  ap.add_argument('-csv', '--use_csv',type=int, default=1, help='1 to use csv, 0 to not. Default is 1')    
  ap.add_argument('-t', '--threads',   type=int, default=1,        help='Number of threads. Default is 1')
  ap.add_argument('-m', '--model',     type=str, default='customcnn.xmodel', help='Path of xmodel. Default is customcnn.xmodel')
  ap.add_argument('-b', '--batch',      type=int, default=0, help='Batch Size, Default is the max that the DPU can handle (0)')
  ap.add_argument('-n', '--model_name', type=str, default="ResNet50", help='Model name, Default is ResNet50')
  ap.add_argument('-l', '--log_file',     type=str, default=None, help='Path and name of log file. Default is logs/convert_{model_name}_T{num_threads}_B{batch_size}.log')

  args = ap.parse_args()  
  
  print ('Command line options:')
  print (' --image_dir  : ', args.image_dir)
  print (' --csv_zip_dir: ', args.csv_zip_dir)
  print (' --use_csv    : ', args.use_csv)
  print (' --threads    : ', args.threads)
  print (' --model      : ', args.model)
  print (' --batch      : ', args.batch)
  print (' --model_name : ', args.model_name)
  print (' --log_file   : ', args.log_file)
  
  print(divider)
  if(args.log_file == None):
     os.makedirs("logs", exist_ok=True)
     logging.basicConfig(filename='logs/FPGA_{}_T{}_B{}.log'.format(args.model_name, args.threads, args.batch), level=logging.INFO)
  else:
     logging.basicConfig(filename=args.log_file, level=logging.INFO)
  
  logging.info('Command line options:')
  logging.info(' --image_dir  : {}'.format(args.image_dir))
  logging.info(' --csv_zip_dir: {}'.format(args.csv_zip_dir))
  logging.info(' --use_csv    : {}'.format(args.use_csv))
  logging.info(' --threads    : {}'.format(args.threads))
  logging.info(' --model      : {}'.format(args.model))
  logging.info(' --batch      : {}'.format(args.batch))
  logging.info(' --model_name : {}'.format(args.model_name))
  logging.info(' --log_file   : {}'.format(args.log_file))


  global_start_time = timer()
  app(args.image_dir,args.csv_zip_dir,args.use_csv,args.threads,args.model,args.batch,args.model_name)
  global_end_time = timer()
  logging.info("Execution Time: %.3f" %(global_end_time - global_start_time))

if __name__ == '__main__':
  main()

