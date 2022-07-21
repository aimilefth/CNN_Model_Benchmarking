#!/bin/sh
# Common
export MODEL_ZIP=Model_zip
export IMAGENET_DATASET=ImageNet_val_1000
export CIFAR_DATASET=cifar_test_1000
export LENET=${MODEL_ZIP}/LeNet5_Cifar10_47pct_0_5MF
export MOBILENET=${MODEL_ZIP}/MobileNetV1_ImageNet_69_87_1_15GF
export RESNET50=${MODEL_ZIP}/ResNet50_ImageNet_70_90_7_76GF
export INCEPTIONV4=${MODEL_ZIP}/InceptionV4_50ep_24_6GF
export RESNET152=${MODEL_ZIP}/ResNet152_trick_35ep_41_9GF

python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${MOBILENET} -b 1 -t FP32
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${MOBILENET} -b 2 -t FP32
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${MOBILENET} -b 4 -t FP32
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${MOBILENET} -b 8 -t FP32
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${MOBILENET} -b 16 -t FP32
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${MOBILENET} -b 32 -t FP32
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${MOBILENET} -b 64 -t FP32
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${MOBILENET} -b 128 -t FP32
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${MOBILENET} -b 256 -t FP32
rm -rf ${MOBILENET}_TensorRT_FP32_BATCH_1
rm -rf ${MOBILENET}_TensorRT_FP32_BATCH_2
rm -rf ${MOBILENET}_TensorRT_FP32_BATCH_4
rm -rf ${MOBILENET}_TensorRT_FP32_BATCH_8
rm -rf ${MOBILENET}_TensorRT_FP32_BATCH_16
rm -rf ${MOBILENET}_TensorRT_FP32_BATCH_32
rm -rf ${MOBILENET}_TensorRT_FP32_BATCH_64
rm -rf ${MOBILENET}_TensorRT_FP32_BATCH_128
rm -rf ${MOBILENET}_TensorRT_FP32_BATCH_256
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${MOBILENET} -b 1 -t FP16
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${MOBILENET} -b 2 -t FP16
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${MOBILENET} -b 4 -t FP16
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${MOBILENET} -b 8 -t FP16
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${MOBILENET} -b 16 -t FP16
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${MOBILENET} -b 32 -t FP16
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${MOBILENET} -b 64 -t FP16
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${MOBILENET} -b 128 -t FP16
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${MOBILENET} -b 256 -t FP16
rm -rf ${MOBILENET}_TensorRT_FP16_BATCH_1
rm -rf ${MOBILENET}_TensorRT_FP16_BATCH_2
rm -rf ${MOBILENET}_TensorRT_FP16_BATCH_4
rm -rf ${MOBILENET}_TensorRT_FP16_BATCH_8
rm -rf ${MOBILENET}_TensorRT_FP16_BATCH_16
rm -rf ${MOBILENET}_TensorRT_FP16_BATCH_32
rm -rf ${MOBILENET}_TensorRT_FP16_BATCH_64
rm -rf ${MOBILENET}_TensorRT_FP16_BATCH_128
rm -rf ${MOBILENET}_TensorRT_FP16_BATCH_256
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${MOBILENET} -b 1 -t INT8
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${MOBILENET} -b 2 -t INT8
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${MOBILENET} -b 4 -t INT8
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${MOBILENET} -b 8 -t INT8
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${MOBILENET} -b 16 -t INT8
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${MOBILENET} -b 32 -t INT8
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${MOBILENET} -b 64 -t INT8
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${MOBILENET} -b 128 -t INT8
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${MOBILENET} -b 256 -t INT8
rm -rf ${MOBILENET}_TensorRT_INT8_BATCH_1
rm -rf ${MOBILENET}_TensorRT_INT8_BATCH_2
rm -rf ${MOBILENET}_TensorRT_INT8_BATCH_4
rm -rf ${MOBILENET}_TensorRT_INT8_BATCH_8
rm -rf ${MOBILENET}_TensorRT_INT8_BATCH_16
rm -rf ${MOBILENET}_TensorRT_INT8_BATCH_32
rm -rf ${MOBILENET}_TensorRT_INT8_BATCH_64
rm -rf ${MOBILENET}_TensorRT_INT8_BATCH_128
rm -rf ${MOBILENET}_TensorRT_INT8_BATCH_256

python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${RESNET50} -b 1 -t FP32
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${RESNET50} -b 2 -t FP32
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${RESNET50} -b 4 -t FP32
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${RESNET50} -b 8 -t FP32
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${RESNET50} -b 16 -t FP32
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${RESNET50} -b 32 -t FP32
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${RESNET50} -b 64 -t FP32
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${RESNET50} -b 128 -t FP32
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${RESNET50} -b 256 -t FP32
rm -rf ${RESNET50}_TensorRT_FP32_BATCH_1
rm -rf ${RESNET50}_TensorRT_FP32_BATCH_2
rm -rf ${RESNET50}_TensorRT_FP32_BATCH_4
rm -rf ${RESNET50}_TensorRT_FP32_BATCH_8
rm -rf ${RESNET50}_TensorRT_FP32_BATCH_16
rm -rf ${RESNET50}_TensorRT_FP32_BATCH_32
rm -rf ${RESNET50}_TensorRT_FP32_BATCH_64
rm -rf ${RESNET50}_TensorRT_FP32_BATCH_128
rm -rf ${RESNET50}_TensorRT_FP32_BATCH_256
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${RESNET50} -b 1 -t FP16
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${RESNET50} -b 2 -t FP16
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${RESNET50} -b 4 -t FP16
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${RESNET50} -b 8 -t FP16
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${RESNET50} -b 16 -t FP16
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${RESNET50} -b 32 -t FP16
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${RESNET50} -b 64 -t FP16
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${RESNET50} -b 128 -t FP16
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${RESNET50} -b 256 -t FP16
rm -rf ${RESNET50}_TensorRT_FP16_BATCH_1
rm -rf ${RESNET50}_TensorRT_FP16_BATCH_2
rm -rf ${RESNET50}_TensorRT_FP16_BATCH_4
rm -rf ${RESNET50}_TensorRT_FP16_BATCH_8
rm -rf ${RESNET50}_TensorRT_FP16_BATCH_16
rm -rf ${RESNET50}_TensorRT_FP16_BATCH_32
rm -rf ${RESNET50}_TensorRT_FP16_BATCH_64
rm -rf ${RESNET50}_TensorRT_FP16_BATCH_128
rm -rf ${RESNET50}_TensorRT_FP16_BATCH_256
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${RESNET50} -b 1 -t INT8
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${RESNET50} -b 2 -t INT8
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${RESNET50} -b 4 -t INT8
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${RESNET50} -b 8 -t INT8
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${RESNET50} -b 16 -t INT8
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${RESNET50} -b 32 -t INT8
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${RESNET50} -b 64 -t INT8
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${RESNET50} -b 128 -t INT8
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${RESNET50} -b 256 -t INT8
rm -rf ${RESNET50}_TensorRT_INT8_BATCH_1
rm -rf ${RESNET50}_TensorRT_INT8_BATCH_2
rm -rf ${RESNET50}_TensorRT_INT8_BATCH_4
rm -rf ${RESNET50}_TensorRT_INT8_BATCH_8
rm -rf ${RESNET50}_TensorRT_INT8_BATCH_16
rm -rf ${RESNET50}_TensorRT_INT8_BATCH_32
rm -rf ${RESNET50}_TensorRT_INT8_BATCH_64
rm -rf ${RESNET50}_TensorRT_INT8_BATCH_128
rm -rf ${RESNET50}_TensorRT_INT8_BATCH_256

python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${LENET} -b 1 -t FP32 -d ${CIFAR_DATASET}
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${LENET} -b 2 -t FP32 -d ${CIFAR_DATASET}
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${LENET} -b 4 -t FP32 -d ${CIFAR_DATASET}
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${LENET} -b 8 -t FP32 -d ${CIFAR_DATASET}
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${LENET} -b 16 -t FP32 -d ${CIFAR_DATASET}
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${LENET} -b 32 -t FP32 -d ${CIFAR_DATASET}
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${LENET} -b 64 -t FP32 -d ${CIFAR_DATASET}
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${LENET} -b 128 -t FP32 -d ${CIFAR_DATASET}
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${LENET} -b 256 -t FP32 -d ${CIFAR_DATASET}
rm -rf ${LENET}_TensorRT_FP32_BATCH_1
rm -rf ${LENET}_TensorRT_FP32_BATCH_2
rm -rf ${LENET}_TensorRT_FP32_BATCH_4
rm -rf ${LENET}_TensorRT_FP32_BATCH_8
rm -rf ${LENET}_TensorRT_FP32_BATCH_16
rm -rf ${LENET}_TensorRT_FP32_BATCH_32
rm -rf ${LENET}_TensorRT_FP32_BATCH_64
rm -rf ${LENET}_TensorRT_FP32_BATCH_128
rm -rf ${LENET}_TensorRT_FP32_BATCH_256
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${LENET} -b 1 -t FP16 -d ${CIFAR_DATASET}
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${LENET} -b 2 -t FP16 -d ${CIFAR_DATASET}
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${LENET} -b 4 -t FP16 -d ${CIFAR_DATASET}
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${LENET} -b 8 -t FP16 -d ${CIFAR_DATASET}
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${LENET} -b 16 -t FP16 -d ${CIFAR_DATASET}
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${LENET} -b 32 -t FP16 -d ${CIFAR_DATASET}
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${LENET} -b 64 -t FP16 -d ${CIFAR_DATASET}
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${LENET} -b 128 -t FP16 -d ${CIFAR_DATASET}
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${LENET} -b 256 -t FP16 -d ${CIFAR_DATASET}
rm -rf ${LENET}_TensorRT_FP16_BATCH_1
rm -rf ${LENET}_TensorRT_FP16_BATCH_2
rm -rf ${LENET}_TensorRT_FP16_BATCH_4
rm -rf ${LENET}_TensorRT_FP16_BATCH_8
rm -rf ${LENET}_TensorRT_FP16_BATCH_16
rm -rf ${LENET}_TensorRT_FP16_BATCH_32
rm -rf ${LENET}_TensorRT_FP16_BATCH_64
rm -rf ${LENET}_TensorRT_FP16_BATCH_128
rm -rf ${LENET}_TensorRT_FP16_BATCH_256
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${LENET} -b 1 -t INT8 -d ${CIFAR_DATASET}
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${LENET} -b 2 -t INT8 -d ${CIFAR_DATASET}
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${LENET} -b 4 -t INT8 -d ${CIFAR_DATASET}
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${LENET} -b 8 -t INT8 -d ${CIFAR_DATASET}
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${LENET} -b 16 -t INT8 -d ${CIFAR_DATASET}
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${LENET} -b 32 -t INT8 -d ${CIFAR_DATASET}
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${LENET} -b 64 -t INT8 -d ${CIFAR_DATASET}
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${LENET} -b 128 -t INT8 -d ${CIFAR_DATASET}
python3 -u GPU_Inference_TensorRT_v2_batch_auto.py -m ${LENET} -b 256 -t INT8 -d ${CIFAR_DATASET}
rm -rf ${LENET}_TensorRT_INT8_BATCH_1
rm -rf ${LENET}_TensorRT_INT8_BATCH_2
rm -rf ${LENET}_TensorRT_INT8_BATCH_4
rm -rf ${LENET}_TensorRT_INT8_BATCH_8
rm -rf ${LENET}_TensorRT_INT8_BATCH_16
rm -rf ${LENET}_TensorRT_INT8_BATCH_32
rm -rf ${LENET}_TensorRT_INT8_BATCH_64
rm -rf ${LENET}_TensorRT_INT8_BATCH_128
rm -rf ${LENET}_TensorRT_INT8_BATCH_256

