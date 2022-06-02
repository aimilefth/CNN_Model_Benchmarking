\# CNN_Model_Benchmarking

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
LeNet5 |Cifar 10| 20 | 48.6 | 0.84 MF |
MobileNetV3_small |ImageNet mini| 62.3 | 84.4 | 92.1 MF
MobileNetV3_large |ImageNet mini| 73.6 | 91.6 | 0.45 GF
ResNet50 |ImageNet mini|70.3 | 90.1 | 7.76 GF |
NASNet_Large |ImageNet mini| 81.3 | 95.7 | 47.8 GF |
MobileNetV1 | Imagenet mini| 68.9 | 88.7 | 1.15 GF |
ResNetV2152 | No training  | -    | -    | 40.5 GF |
ResNet152   | 35 Epochs ImageNet mini| -| 20 | 41.9 GF|
InceptionV4 | 50 Epochs ImageNet mini| -| -  | 24.6 GF|

Server Time (ms) Intel(R) Xeon(R) Silver 4210 CPU @ 2.20GHz

Threads |LeNet5| MobileNetV3_small | MobileNetV3_large| MobileNetV1 | ResNet50 | InceptionV4  | ResNetV2152 | ResNet152 | NASNet_large |
---  | ---:    | ---:    | ---:     | ---:     | ---:     | ---:      | ---:      | ---:      |  ---:     |
None |0.665    |8.883    |24.955    |21.286    |112.377   |287.695    |401.403    |473.781    |1048.949   |
1    |**0.393**|**6.596**|24.499    |42.800    |253.742   |752.708    |1240.437   |1242.706   |1703.825   |
2    |0.601    |8.994    |27.335    |28.638    |180.110   |490.097    |658.236    |809.723    |1303.018   |
4    |0.698    |8.820    |21.444    |23.660    |116.357   |294.695    |415.772    |473.536    |915.699    |
8    |0.645    |8.461    |**20.114**|19.240    |82.583    |209.656    |271.152    |316.587    |708.663    |
16   |0.565    |7.980    |21.791    |15.125    |69.426    |165.007    |**236.015**|**258.137**|626.483    |
32   |0.587    |8.015    |22.025    |**14.623**|**63.731**|**164.348**|251.068    |285.480    |**614.677**|
64   |0.655    |8.698    |22.131    |15.649    |71.173    |169.745    |250.925    |278.666    |630.602    |

<!---
docker run -it --rm --name tf_intel --mount src=~/Documents/CNN_Model_Benchmarking,target=/home/aimleft,type=bind -p 8888:8888 -p 6006:6006 intel/intel-optimized-tensorflow

docker run -it --rm --name tf_arm --mount src=/home/petalinux/CNN_Model_Benchmarking,target=/home/aimleft,type=bind -p 8888:8888 -p 6006:6006 armswdev/tensorflow-arm-neoverse:latest

-->

ARM Time (ms) quad-core ARM Cortex-A53 @1.2GHz

Threads |LeNet5| MobileNetV3_small | MobileNetV3_large| MobileNetV1 | ResNet50 | InceptionV4  | ResNetV2152 | ResNet152 | NASNet_large |
---  | ---:    | ---:     | ---:      | ---:      | ---:      | ---:       | ---:      | ---:       |  ---:      |
None |1.580    |36.789    |131.900    |344.507    |1456.592   |4196.161    |7156.722   |7289.745    |12234.435   |
1    |1.599    |36.883    |130.941    |342.899    |1448.184   |4195.403    |7149.648   |7285.386    |12226.411   |
2    |1.495    |20.970    |71.134     |185.353    |785.006    |2235.095    |3804.390   |3882.866    |6594.041    |
4    |**1.457**|**13.353**|**43.302** |**106.830**|**450.416**|**1211.299**|**2141.35**|**2174.237**|**3821.893**|

ARM Time (ms) quad-core ARM Cortex-A53 @1.2GHz uint8 quantized

Threads |LeNet5| MobileNetV3_small | MobileNetV3_large| MobileNetV1 | ResNet50 | InceptionV4  | ResNetV2152 | ResNet152 | NASNet_large |
---  | ---:    | ---:     | ---:      | ---:     | ---:      | ---:       | ---:       | ---:       |  ---:      |
None |1.616    |46.941    |146.420    |220.116   |1264.388   |3701.993    |6515.394    |6427.714    |9291.822    |
1    |1.608    |47.077    |146.492    |220.861   |1264.796   |3703.152    |6516.713    |6429.290    |9303.939    |
2    |1.507    |30.496    |88.222     |124.742   |677.333    |1944.893    |3565.661    |3354.058    |5191.676    |
4    |**1.438**|**23.279**|**59.287** |**76.343**|**380.693**|**1070.666**|**2070.767**|**1834.977**|**3139.608**|


<!---
GPU Time (ms) Tesla P100-PCIE-16gb

MODE |LeNet5| MobileNetV3_small | MobileNetV3_large| ResNet50 | NASNet_large |
---  | ---:    | ---:     | ---:      | ---:      |  ---:     |
Transfering inputs in one go |1.023 |14.470|20.841|23.235|94.629 |
Transfering inputs one by one|33.687|44.547|47.349|48.377|124.676|


GPU Time (ms) Tesla V100S PCIE-32gb

MODE |LeNet5| MobileNetV3_small | MobileNetV3_large| ResNet50 | NASNet_large |
---  | ---:    | ---:     | ---:      | ---:      |  ---:     |
Transfering inputs in one go | 2.975 | 36.331 | 39.778 | 39.659 |110.298
Transfering inputs one by one|64.128 | 98.906 | 100.672| 97.908 |203.269
-->

GPU Time (ms) Tesla V100S PCIE-32gb NVIDIA-SMI 460.80 CUDA Version: 11.2

MODE |LeNet5| MobileNetV3_small | MobileNetV3_large| MobileNetV1 | ResNet50 | InceptionV4  | ResNetV2152 | ResNet152 | NASNet_large |
---  | ---: | ---: | ---: | ---: | ---: | ---: | ---:|---: | ---:|
model call for loop              |    2.384|   60.641|   74.420|   27.205|   48.178|  123.307|  146.004|  145.413|   329.312|
model.predict() with batch_size=1|    1.627|   24.088|   26.278|   14.002|   21.232|   52.220|   48.924|   50.354|    98.247|
model call for loop w/jit compile|    0.494|    2.644|    3.626|    1.756|    3.840|    8.636|    8.197|    9.939|    20.556|
TensorRT                         |**0.445**|**1.975**|**2.095**|**1.051**|**2.275**|**6.663**|**6.733**|**4.100**|**15.082**|
1st call of model.predict()      | 1059.716| 1673.415| 1703.903| 1875.492| 1652.621| 2041.034| 3807.492| 1467.161|  1636.244|
1st call of model call w/jit     |  543.791| 3046.853| 3402.673| 3508.573| 2122.300| 9094.371| 6456.305| 6949.572| 13989.758|
1st call of TensorRT             |   14.743|  144.914|  262.083|  185.708|  921.957| 1775.576|  622.452| 2475.927|  4317.527|


GPU Time (ms) Tesla V100S PCIE-32gb NVIDIA-SMI 460.80 CUDA Version: 11.2 FP16

MODE |LeNet5| MobileNetV3_small | MobileNetV3_large| MobileNetV1 | ResNet50 | InceptionV4  | ResNetV2152 | ResNet152 | NASNet_large |
---  | ---: | ---: | ---: | ---: | ---: | ---: | ---:|---: | ---:|
TensorRT                         |0.381 |2.241  |2.448  |1.328  |1.927  |   4.192|   5.572|   5.248|  13.380|
1st call of TensorRT             |16.433|145.526|260.503|170.576|884.686|1778.258|2268.147|2405.281|4370.088|


GPU Time (ms) Tesla V100S PCIE-32gb NVIDIA-SMI 460.80 CUDA Version: 11.2 INT8 Quantized

MODE |LeNet5| MobileNetV3_small | MobileNetV3_large| MobileNetV1 | ResNet50 | InceptionV4  | ResNetV2152 | ResNet152 | NASNet_large |
---  | ---: | ---: | ---: | ---: | ---: | ---: | ---:|---: | ---:|
TensorRT                         |0.576 |2.493  |2.727  |1.252  |2.334  |   4.414|   5.698|   5.603|  14.689|
1st call of TensorRT             |13.170|151.099|270.721|173.666|966.042|1911.517|2438.974|2380.068|4307.959|

No perfomance increase because of no INT8 tensor cores on Volta GPUs
nvidia.com/en-us/data-center/tensor-cores/

All the averages exclude the 1st call in the calculation

TensorRT with tensorflow/tensorflow:latest-gpu (TensorRT 7.2.2)

Rest are with nvcr.io/nvidia/tensorflow:22.03-tf2-py3 (TensorRT 8.2.3)

<!---

docker run -it --rm --name tf_tensorrt --gpus all --mount src=~/Documents/CNN_Model_Benchmarking,target=/home/aimleft,type=bind -p 8888:8888 -p 6006:6006 nvcr.io/nvidia/tensorflow:22.03-tf2-py3

GPU Time (ms) Tesla P100 PCIE-16gb NVIDIA-SMI 470.82.01 CUDA Version: 11.4 Kaggle

MODE |LeNet5| MobileNetV3_small | MobileNetV3_large | ResNet50 | NASNet_large |
---  | ---: | ---: | ---: | ---: | ---: |
model call for loop              |    2.131|   58.753|   65.638|   48.983|   304.902|
model.predict() with batch_size=1|    1.105|   17.302|   21.858|   21.209|    87.984|
model call for loop w/jit compile|    0.577|    4.381|    5.736|    7.247|    41.253|
1st call of model.predict()      | 4793.978| 5836.142| 5726.507| 5956.792|  5775.364|
1st call of model call w/jit     |  684.943| 1567.411| 3390.069| 2503.253| 15953.979|
-->

FPGA time (ms) using Vitis AI 2.0 on ZCU104 with a 2 * B4096 @ 300MHz DPU configuration

LeNet5| MobileNetV1| ResNet50 | InceptionV4 | ResNet152
 ---:    | ---:    | ---:     | ---:        |  ---:     |
0.200    |2.676    |11.442    |33.731       |58.548     | 

FPGA time (ms) using Vitis AI 1.4.1 on U280 with 2 DPUCAHX8L kernels running at 250Mhz

LeNet5| MobileNetV1| ResNet50 | InceptionV4 | ResNet152
 ---:    | ---:    | ---:     | ---:        |  ---:     |
0.217    |1.141    |12.745    |19.464       |46.945     | 

<!---
docker run -it --rm --name tf_tensorrt --runtime nvidia --mount src=~/Documents/CNN_Model_Benchmarking,target=/home/aimleft,type=bind --network host nvcr.io/nvidia/l4t-tensorflow:r32.5.0-tf2.3-py3
-->
Jetson Nano nx02 (ms) using TensorRT, FP32

MODE |LeNet5| MobileNetV1 | ResNet50 | InceptionV4  | ResNet152 |
---  | ---: | ---: | ---: | ---: | ---: |
TensorRT            | | | | | |
1st call of TensorRT| | | | | |

