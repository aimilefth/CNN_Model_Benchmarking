# CNN_Model_Benchmarking

TF 2.6

Each Folder Contains 
1) a ipynb file to train the model and to test the inference time
2) a zip containing the saved model in SavedModel format 
   OR a .txt file as a placeholder, because the model file is too big (>50MB)


The name of each saved model has the following format:
<name>_<dataset>_<metric>_<FLOPS>
  
Currently only Image Classification Tasks have been created
  
Architecture | Dataset| Top1 Accuracy | Top5 Accuracy|  FLOPS |
--- | --- | ---: | --- | ---:| 
LeNet5 |Cifar 10| 20 | 47.1 | 0.5 MF |
MobileNetV3_small |ImageNet mini| 62.3 | 84.4 | 92.1 MF
MobileNetV3_large |ImageNet mini| 73.6 | 91.6 | 0.23 GF
ResNet50 |ImageNet mini|70.3 | 90.1 | 7.76 GF |
NASNet_Large |ImageNet mini| 81.3 | 95.7 | 47.8 GF |
