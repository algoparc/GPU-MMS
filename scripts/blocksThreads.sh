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

# Runs GPU-MMS for varying numbers of threads per block and blocks per grid.  Used to determine how performance
#varies with these parameters.
# NOTE: Change the compute capability in the nvcc commands below based on the GPU you are using.

#!/bin/bash

for t in 64 128 256
do
  for b in 64 128 256
  do
nvcc -std=c++11 -O3 -DTHREADS=$t -DBLOCKS=$b -use_fast_math --expt-extended-lambda -Xptxas -dlcm=cg --maxrregcount=60 -lcudart -D_FORCE_INLINES -gencode arch=compute_30,code=sm_30 -c testMM.cu
nvcc testMM.o -std=c++11 -O3 -DTHREADS=$t -DBLOCKS=$b -use_fast_math --expt-extended-lambda -Xptxas -dlcm=cg --maxrregcount=60 -lcudart -D_FORCE_INLINES -gencode arch=compute_30,code=sm_30 -o testMM
  ./testMM
  done
  echo ""
done  
