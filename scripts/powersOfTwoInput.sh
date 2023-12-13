
#  (C) Copyright 2016-2018 Ben Karsin, Nodari Sitchinava
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *     http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# *


#!/bin/bash

# Runs GPU-MMS on a series of different size random inputs.  This script is used to see how GPU-MMS performance
# varies with input size, N
# NOTE: Change the compute capability in the nvcc commands below based on the GPU you are using.

for n in 2097152 4194304 8388608 16777216 33554432 67108864 134217728
do
    nvcc -std=c++11 -O3  -lineinfo -DN=${n} -use_fast_math --maxrregcount=60 --expt-extended-lambda -Xptxas -dlcm=cg -lcudart -D_FORCE_INLINES -gencode arch=compute_30,code=sm_30 -c testMM.cu
nvcc testMM.o -std=c++11 -O3  -lineinfo -use_fast_math --maxrregcount=60 --expt-extended-lambda -Xptxas -dlcm=cg -lcudart -D_FORCE_INLINES -gencode arch=compute_30,code=sm_30 -o testMM
    ./testMM 
  echo ""
done
