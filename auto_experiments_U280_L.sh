#!/bin/sh

export TARGET=target_U280_L
export LENET5=./LeNet5/${TARGET}/customcnn.xmodel
export MOBILENETV1=./MobileNetV1/${TARGET}/customcnn.xmodel
export RESNET50=./ResNet50/${TARGET}/customcnn.xmodel
export INCEPTIONV4=./InceptionV4/${TARGET}/customcnn.xmodel
export RESNET152=./ResNet152/${TARGET}/customcnn.xmodel
export IMAGENET=ImageNet_val_100.zip
export CIFAR=cifar_test_1000.zip

python3 -u app_mt_v2_test_log.py -n MobileNetV1 -z ${IMAGENET} -m ${MOBILENETV1} -b 1 -t 1
python3 -u app_mt_v2_test_log.py -n MobileNetV1 -z ${IMAGENET} -m ${MOBILENETV1} -b 1 -t 2 
python3 -u app_mt_v2_test_log.py -n MobileNetV1 -z ${IMAGENET} -m ${MOBILENETV1} -b 1 -t 3 
python3 -u app_mt_v2_test_log.py -n MobileNetV1 -z ${IMAGENET} -m ${MOBILENETV1} -b 1 -t 4
python3 -u app_mt_v2_test_log.py -n MobileNetV1 -z ${IMAGENET} -m ${MOBILENETV1} -b 1 -t 8

python3 -u app_mt_v2_test_log.py -n ResNet50 -z ${IMAGENET} -m ${RESNET50} -b 1 -t 1
python3 -u app_mt_v2_test_log.py -n ResNet50 -z ${IMAGENET} -m ${RESNET50} -b 1 -t 2
python3 -u app_mt_v2_test_log.py -n ResNet50 -z ${IMAGENET} -m ${RESNET50} -b 1 -t 3
python3 -u app_mt_v2_test_log.py -n ResNet50 -z ${IMAGENET} -m ${RESNET50} -b 1 -t 4
python3 -u app_mt_v2_test_log.py -n ResNet50 -z ${IMAGENET} -m ${RESNET50} -b 1 -t 8

python3 -u app_mt_v2_test_log.py -n InceptionV4 -z ${IMAGENET} -m ${INCEPTIONV4} -b 1 -t 1
python3 -u app_mt_v2_test_log.py -n InceptionV4 -z ${IMAGENET} -m ${INCEPTIONV4} -b 1 -t 2
python3 -u app_mt_v2_test_log.py -n InceptionV4 -z ${IMAGENET} -m ${INCEPTIONV4} -b 1 -t 3
python3 -u app_mt_v2_test_log.py -n InceptionV4 -z ${IMAGENET} -m ${INCEPTIONV4} -b 1 -t 4
python3 -u app_mt_v2_test_log.py -n InceptionV4 -z ${IMAGENET} -m ${INCEPTIONV4} -b 1 -t 8

python3 -u app_mt_v2_test_log.py -n ResNet152 -z ${IMAGENET} -m ${RESNET152} -b 1 -t 1
python3 -u app_mt_v2_test_log.py -n ResNet152 -z ${IMAGENET} -m ${RESNET152} -b 1 -t 2
python3 -u app_mt_v2_test_log.py -n ResNet152 -z ${IMAGENET} -m ${RESNET152} -b 1 -t 3
python3 -u app_mt_v2_test_log.py -n ResNet152 -z ${IMAGENET} -m ${RESNET152} -b 1 -t 4
python3 -u app_mt_v2_test_log.py -n ResNet152 -z ${IMAGENET} -m ${RESNET152} -b 1 -t 8

python3 -u app_mt_v2_test_log.py -n LeNet5 -z ${CIFAR} -m ${LENET5} -b 1 -t 1
python3 -u app_mt_v2_test_log.py -n LeNet5 -z ${CIFAR} -m ${LENET5} -b 1 -t 2
python3 -u app_mt_v2_test_log.py -n LeNet5 -z ${CIFAR} -m ${LENET5} -b 1 -t 3
python3 -u app_mt_v2_test_log.py -n LeNet5 -z ${CIFAR} -m ${LENET5} -b 1 -t 4
python3 -u app_mt_v2_test_log.py -n LeNet5 -z ${CIFAR} -m ${LENET5} -b 1 -t 8
