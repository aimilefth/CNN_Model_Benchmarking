#!/bin/sh

# Copyright 2020 Xilinx Inc.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Author: Mark Harvey, Xilinx Inc

if [ $1 = zcu102 ]; then
      ARCH=/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU102/arch.json
      TARGET=zcu102
      echo "-----------------------------------------"
      echo "COMPILING MODEL FOR ZCU102.."
      echo "-----------------------------------------"
elif [ $1 = zcu104 ]; then
      ARCH=/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json
      TARGET=zcu104
      echo "-----------------------------------------"
      echo "COMPILING MODEL FOR ZCU104.."
      echo "-----------------------------------------"
elif [ $1 = vck190 ]; then
      ARCH=/opt/vitis_ai/compiler/arch/DPUCVDX8G/VCK190/arch.json
      TARGET=vck190
      echo "-----------------------------------------"
      echo "COMPILING MODEL FOR VCK190.."
      echo "-----------------------------------------"
elif [ $1 = u50_H ]; then
      ARCH=/opt/vitis_ai/compiler/arch/DPUCAHX8H/U50/arch.json
      TARGET=u50_H
      echo "-----------------------------------------"
      echo "COMPILING MODEL FOR ALVEO U50_H.."
      echo "-----------------------------------------"
elif [ $1 = u50_L ]; then
      ARCH=/opt/vitis_ai/compiler/arch/DPUCAHX8L/U50/arch.json
      TARGET=u50_L
      echo "-----------------------------------------"
      echo "COMPILING MODEL FOR ALVEO U50_L.."
      echo "-----------------------------------------"
elif [ $1 = u280_H ]; then
      ARCH=/opt/vitis_ai/compiler/arch/DPUCAHX8H/U280/arch.json
      TARGET=u280_H
      echo "-----------------------------------------"
      echo "COMPILING MODEL FOR ALVEO U280_H.."
      echo "-----------------------------------------"
elif [ $1 = u280_L ]; then
      ARCH=/opt/vitis_ai/compiler/arch/DPUCAHX8L/U280/arch.json
      TARGET=u280_L
      echo "-----------------------------------------"
      echo "COMPILING MODEL FOR ALVEO U280_L.."
      echo "-----------------------------------------"
elif [ $1 = u200 ]; then
      ARCH=/opt/vitis_ai/compiler/arch/DPUCADF8H/U200/arch.json
      TARGET=u200
      echo "-----------------------------------------"
      echo "COMPILING MODEL FOR ALVEO U200.."
      echo "-----------------------------------------"      
else
      echo  "Target not found. Valid choices are: zcu102, zcu104, vck190, u50_H, u50_L, u280_H, u280_L, U200 ..exiting"
      exit 1
fi

compile() {
      vai_c_tensorflow2 \
            --model           build/quant_model/q_model.h5 \
            --arch            $ARCH \
            --output_dir      build/compiled_$TARGET \
            --net_name        customcnn
}


compile 2>&1 | tee build/logs/compile_$TARGET.log


echo "-----------------------------------------"
echo "MODEL COMPILED"
echo "-----------------------------------------"



