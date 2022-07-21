#!/bin/sh
# Common
export IMAGENET_DATASET=ImageNet_val_1000
export CIFAR_DATASET=cifar_test_1000
export LENET_TFLITE=LeNet5/LeNet5_Cifar10_47pct_0_5MF.tflite
export MOBILENET_TFLITE=MobileNetV1/MobileNetV1_ImageNet_69_87_1_15GF.tflite
export RESNET50_TFLITE=ResNet50/ResNet50_ImageNet_70_90_7_76GF.tflite
export INCEPTIONV4_TFLITE=InceptionV4/InceptionV4_50ep_24_6GF.tflite
export RESNET152_TFLITE=ResNet152/ResNet152_trick_35ep_41_9GF.tflite
: '
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${MOBILENET_TFLITE} -b 1 -t 1
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${MOBILENET_TFLITE} -b 2 -t 1
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${MOBILENET_TFLITE} -b 4 -t 1
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${MOBILENET_TFLITE} -b 8 -t 1
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${MOBILENET_TFLITE} -b 16 -t 1
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${MOBILENET_TFLITE} -b 32 -t 1
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${MOBILENET_TFLITE} -b 64 -t 1
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${MOBILENET_TFLITE} -b 128 -t 1
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${MOBILENET_TFLITE} -b 256 -t 1
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${MOBILENET_TFLITE} -b 1 -t 4
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${MOBILENET_TFLITE} -b 2 -t 4
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${MOBILENET_TFLITE} -b 4 -t 4
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${MOBILENET_TFLITE} -b 8 -t 4
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${MOBILENET_TFLITE} -b 16 -t 4
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${MOBILENET_TFLITE} -b 32 -t 4
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${MOBILENET_TFLITE} -b 64 -t 4
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${MOBILENET_TFLITE} -b 128 -t 4
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${MOBILENET_TFLITE} -b 256 -t 4
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${MOBILENET_TFLITE} -b 1 -t 16
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${MOBILENET_TFLITE} -b 2 -t 16
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${MOBILENET_TFLITE} -b 4 -t 16
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${MOBILENET_TFLITE} -b 8 -t 16
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${MOBILENET_TFLITE} -b 16 -t 16
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${MOBILENET_TFLITE} -b 32 -t 16
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${MOBILENET_TFLITE} -b 64 -t 16
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${MOBILENET_TFLITE} -b 128 -t 16
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${MOBILENET_TFLITE} -b 256 -t 16

python3 -u get_batched_inference_tflite_jpg_auto.py -m ${RESNET50_TFLITE} -b 1 -t 1
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${RESNET50_TFLITE} -b 2 -t 1
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${RESNET50_TFLITE} -b 4 -t 1
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${RESNET50_TFLITE} -b 8 -t 1
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${RESNET50_TFLITE} -b 16 -t 1
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${RESNET50_TFLITE} -b 32 -t 1
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${RESNET50_TFLITE} -b 64 -t 1
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${RESNET50_TFLITE} -b 128 -t 1
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${RESNET50_TFLITE} -b 256 -t 1
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${RESNET50_TFLITE} -b 1 -t 4
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${RESNET50_TFLITE} -b 2 -t 4
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${RESNET50_TFLITE} -b 4 -t 4
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${RESNET50_TFLITE} -b 8 -t 4
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${RESNET50_TFLITE} -b 16 -t 4
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${RESNET50_TFLITE} -b 32 -t 4
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${RESNET50_TFLITE} -b 64 -t 4
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${RESNET50_TFLITE} -b 128 -t 4
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${RESNET50_TFLITE} -b 256 -t 4
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${RESNET50_TFLITE} -b 1 -t 16
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${RESNET50_TFLITE} -b 2 -t 16
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${RESNET50_TFLITE} -b 4 -t 16
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${RESNET50_TFLITE} -b 8 -t 16
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${RESNET50_TFLITE} -b 16 -t 16
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${RESNET50_TFLITE} -b 32 -t 16
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${RESNET50_TFLITE} -b 64 -t 16
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${RESNET50_TFLITE} -b 128 -t 16
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${RESNET50_TFLITE} -b 256 -t 16

python3 -u get_batched_inference_tflite_jpg_auto.py -m ${INCEPTIONV4_TFLITE} -b 1 -t 1
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${INCEPTIONV4_TFLITE} -b 2 -t 1
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${INCEPTIONV4_TFLITE} -b 4 -t 1
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${INCEPTIONV4_TFLITE} -b 8 -t 1
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${INCEPTIONV4_TFLITE} -b 16 -t 1
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${INCEPTIONV4_TFLITE} -b 32 -t 1
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${INCEPTIONV4_TFLITE} -b 64 -t 1
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${INCEPTIONV4_TFLITE} -b 128 -t 1
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${INCEPTIONV4_TFLITE} -b 256 -t 1
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${INCEPTIONV4_TFLITE} -b 1 -t 4
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${INCEPTIONV4_TFLITE} -b 2 -t 4
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${INCEPTIONV4_TFLITE} -b 4 -t 4
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${INCEPTIONV4_TFLITE} -b 8 -t 4
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${INCEPTIONV4_TFLITE} -b 16 -t 4
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${INCEPTIONV4_TFLITE} -b 32 -t 4
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${INCEPTIONV4_TFLITE} -b 64 -t 4
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${INCEPTIONV4_TFLITE} -b 128 -t 4
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${INCEPTIONV4_TFLITE} -b 256 -t 4
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${INCEPTIONV4_TFLITE} -b 1 -t 16
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${INCEPTIONV4_TFLITE} -b 2 -t 16
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${INCEPTIONV4_TFLITE} -b 4 -t 16
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${INCEPTIONV4_TFLITE} -b 8 -t 16
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${INCEPTIONV4_TFLITE} -b 16 -t 16
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${INCEPTIONV4_TFLITE} -b 32 -t 16
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${INCEPTIONV4_TFLITE} -b 64 -t 16
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${INCEPTIONV4_TFLITE} -b 128 -t 16
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${INCEPTIONV4_TFLITE} -b 256 -t 16

python3 -u get_batched_inference_tflite_jpg_auto.py -m ${RESNET152_TFLITE} -b 1 -t 1
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${RESNET152_TFLITE} -b 2 -t 1
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${RESNET152_TFLITE} -b 4 -t 1
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${RESNET152_TFLITE} -b 8 -t 1
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${RESNET152_TFLITE} -b 16 -t 1
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${RESNET152_TFLITE} -b 32 -t 1
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${RESNET152_TFLITE} -b 64 -t 1
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${RESNET152_TFLITE} -b 128 -t 1
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${RESNET152_TFLITE} -b 256 -t 1
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${RESNET152_TFLITE} -b 1 -t 4
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${RESNET152_TFLITE} -b 2 -t 4
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${RESNET152_TFLITE} -b 4 -t 4
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${RESNET152_TFLITE} -b 8 -t 4
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${RESNET152_TFLITE} -b 16 -t 4
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${RESNET152_TFLITE} -b 32 -t 4
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${RESNET152_TFLITE} -b 64 -t 4
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${RESNET152_TFLITE} -b 128 -t 4
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${RESNET152_TFLITE} -b 256 -t 4
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${RESNET152_TFLITE} -b 1 -t 16
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${RESNET152_TFLITE} -b 2 -t 16
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${RESNET152_TFLITE} -b 4 -t 16
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${RESNET152_TFLITE} -b 8 -t 16
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${RESNET152_TFLITE} -b 16 -t 16
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${RESNET152_TFLITE} -b 32 -t 16
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${RESNET152_TFLITE} -b 64 -t 16
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${RESNET152_TFLITE} -b 128 -t 16
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${RESNET152_TFLITE} -b 256 -t 16
'
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${LENET_TFLITE} -b 1 -t 1 -d ${CIFAR_DATASET}
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${LENET_TFLITE} -b 2 -t 1 -d ${CIFAR_DATASET}
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${LENET_TFLITE} -b 4 -t 1 -d ${CIFAR_DATASET}
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${LENET_TFLITE} -b 8 -t 1 -d ${CIFAR_DATASET}
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${LENET_TFLITE} -b 16 -t 1 -d ${CIFAR_DATASET}
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${LENET_TFLITE} -b 32 -t 1 -d ${CIFAR_DATASET}
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${LENET_TFLITE} -b 64 -t 1 -d ${CIFAR_DATASET}
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${LENET_TFLITE} -b 128 -t 1 -d ${CIFAR_DATASET}
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${LENET_TFLITE} -b 256 -t 1 -d ${CIFAR_DATASET}
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${LENET_TFLITE} -b 1 -t 4 -d ${CIFAR_DATASET}
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${LENET_TFLITE} -b 2 -t 4 -d ${CIFAR_DATASET}
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${LENET_TFLITE} -b 4 -t 4 -d ${CIFAR_DATASET}
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${LENET_TFLITE} -b 8 -t 4 -d ${CIFAR_DATASET}
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${LENET_TFLITE} -b 16 -t 4 -d ${CIFAR_DATASET}
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${LENET_TFLITE} -b 32 -t 4 -d ${CIFAR_DATASET}
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${LENET_TFLITE} -b 64 -t 4 -d ${CIFAR_DATASET}
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${LENET_TFLITE} -b 128 -t 4 -d ${CIFAR_DATASET}
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${LENET_TFLITE} -b 256 -t 4 -d ${CIFAR_DATASET}
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${LENET_TFLITE} -b 1 -t 16 -d ${CIFAR_DATASET}
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${LENET_TFLITE} -b 2 -t 16 -d ${CIFAR_DATASET}
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${LENET_TFLITE} -b 4 -t 16 -d ${CIFAR_DATASET}
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${LENET_TFLITE} -b 8 -t 16 -d ${CIFAR_DATASET}
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${LENET_TFLITE} -b 16 -t 16 -d ${CIFAR_DATASET}
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${LENET_TFLITE} -b 32 -t 16 -d ${CIFAR_DATASET}
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${LENET_TFLITE} -b 64 -t 16 -d ${CIFAR_DATASET}
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${LENET_TFLITE} -b 128 -t 16 -d ${CIFAR_DATASET}
python3 -u get_batched_inference_tflite_jpg_auto.py -m ${LENET_TFLITE} -b 256 -t 16 -d ${CIFAR_DATASET}



