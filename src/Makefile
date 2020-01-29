
#OPTIONS=-std=c++11 -Xcompiler="-Wundef" -O2 -g -Xcompiler="-Werror" -lineinfo  --expt-extended-lambda -use_fast_math -Xptxas="-v" -D_FORCE_INLINES

GENCODE_SM20	:= -gencode arch=compute_20,code=sm_20
GENCODE_SM30	:= -gencode arch=compute_30,code=sm_30
#GENCODE_SM52	:= -gencode arch=compute_52,code=sm_52
GENCODE_SM52	:= -arch compute_52 -code sm_52

#UPDATE THE GENCODE HERE FOR YOUR PARTICULAR HARDWARE

OPTIONS=-std=c++11 -O3  -lineinfo -use_fast_math --expt-extended-lambda -Xptxas -dlcm=cg --maxrregcount=60 -lcudart -D_FORCE_INLINES $(GENCODE_SM30)
NVCC=/usr/local/cuda-10.1/bin/nvcc

default: testMM

testMM: testMM.o
	$(NVCC) testMM.o $(OPTIONS) -o testMM

testMM.o: testMM.cu multimergesort.hxx params.h io-merge.hxx warp-partition.hxx buildData.h basecase/squareSort.hxx io-merge-gen.hxx cmp.hxx basecase/sortRowMajor.hxx
	$(NVCC) $(OPTIONS) -c testMM.cu

#eval-mgpu: eval-mgpu.cu
#	nvcc eval-mgpu.cu -std=c++11 -O3 -arch=sm_30 -D_FORCE_INLINES --expt-extended-lambda -I "../mgpu/src" -o eval-mgpu

clean:
	rm -f *.o testMM
