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
MobileNetV3_large |ImageNet mini| 73.6 | 91.6 | 0.45 GF
ResNet50 |ImageNet mini|70.3 | 90.1 | 7.76 GF |
NASNet_Large |ImageNet mini| 81.3 | 95.7 | 47.8 GF |

Server Time (ms) Intel(R) Xeon(R) Silver 4210 CPU @ 2.20GHz

Threads |LeNet5| MobileNetV3_small | MobileNetV3_large| ResNet50 | NASNet_large |
---  | ---:    | ---:    | ---:     | ---:     |  ---:     |
None |0.665    |8.883    |24.955    |112.377   |1048.949   |
1    |**0.393**|**6.596**|24.499    |253.742   |1703.825   |
2    |0.601    |8.994    |27.335    |180.110   |1303.018   |
4    |0.698    |8.820    |21.444    |116.357   |915.699    |
8    |0.645    |8.461    |**20.114**|82.583    |708.663    |
16   |0.565    |7.980    |21.791    |69.426    |626.483    |
32   |0.587    |8.015    |22.025    |**63.731**|**614.677**|
64   |0.655    |8.698    |22.131    |71.173    |630.602    |


ARM Time (ms) quad-core ARM Cortex-A53 @1.2GHz

Threads |LeNet5| MobileNetV3_small | MobileNetV3_large| ResNet50 | NASNet_large |
---  | ---:    | ---:     | ---:      | ---:      |  ---:     |
None |1.580    |36.789    |131.900    |1456.592   |12234.435   |
1    |1.599    |36.883    |130.941    |1448.184   |12226.411   |
2    |1.495    |20.970    |71.134     |785.006    |6594.041    |
4    |**1.457**|**13.353**|**43.302** |**450.416**|**3821.893**|

GPU Time (ms) Tesla P100-PCIE-16gb

MODE |LeNet5| MobileNetV3_small | MobileNetV3_large| ResNet50 | NASNet_large |
---  | ---:    | ---:     | ---:      | ---:      |  ---:     |
Transfering inputs in one go |1.023 |14.470|20.841|23.235|94.629 |
Transfering inputs one by one|33.687|44.547|47.349|48.377|124.676|

