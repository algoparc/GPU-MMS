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

# Runs GPU-MMS for varying K (branching factor).  Used to determine how performance varies with K.  See
# paper for results for each of our hardware platforms.
# NOTE: Change the compute capability in the nvcc commands below based on the GPU you are using.

#!/bin/bash

k=1
for pl in 1 2 3 4 5 
do
  k=$(($k * 2))
  for threads in 64
  do
    nvcc -std=c++11 -O3  -DK=${k} -DPL=${pl} -DTHREADS=${threads} -use_fast_math --expt-extended-lambda -Xptxas -dlcm=cg -lcudart -D_FORCE_INLINES -gencode arch=compute_52,code=sm_52 -c testMM.cu
nvcc testMM.o -std=c++11 -O3  -use_fast_math --expt-extended-lambda -Xptxas -dlcm=cg -lcudart -D_FORCE_INLINES -gencode arch=compute_52,code=sm_52 -o testMM
    ./testMM 260
  done
  echo ""
done
