# GPU-MMS
Multiway mergesort for GPUs

Developed and implemented by the <a href=http://algoparc.ics.hawaii.edu/>Algorithms and Parallel Computing (AlgoPARC) research group</a> at the University of Hawai'i at Manoa.

This is an implementaton of the sorting algorithm described in the paper: 

B. Karsin, V. Weichert, H. Casanova, J. Iacono, and N. Sitchinava. Analysis-driven Engineering of Comparison-based Sorting Algorithms on GPUs. In Proceedings of the 32nd International Conference on Supercomputing (ICS), 2018, to appear.

More details are available in chapter 7 (case study: sorting) of Ben Karsin's dissertation: https://benkarsin.files.wordpress.com/2018/10/dissertation.pdf

## Using GPU-MMS

src/testMM.cu provides an example of how to call GPU-MMS to sort a given input.  To use GPU-MMS in your own code, include "src/multimergesort.hxx" and call the sorting function by:

> multimergesort<T,cmp>(input, output, host_array, threadBlocks, N);

With the following parameters:
- T: datatype of input being sorted
- cmp: comparison function used to sort by
- input: input array (defined as memory on the GPU)
- output: output array (defined as memory on the GPU)
- host_array: array defined as host memory (used for debugging/correctness verification)
- threadBlocks: number of GPU thread blocks to use
- N: size of array being sorted

In addition to these parameters, a series of parameters that dictate how the GPU-MMS algorithm runs are defined in "src/params.h".  These parameters can be left as-is, or can be changed for users who want to further customize GPU-MMS operation.
- K: Branching factor of GPU-MMS (number of lists merged into one per round)
- PL: Log of K
- PIPELINE: Flag to enable pipelining of merges in the blockHeap
- RANGE: Range of values for randomly generated integer input

## Code Organization

The code is organized as follows:

- src/testMM.cu is a test harness that builds random input sets and calls multimergesort to sort them.
- src/Makefile builds the executable testMM
- src/multimergesort.hxx contains the main sorting method that uses methods from the following files:
- src/basecase/ contains all of the files for the 1024 base case sort.
- src/warp-partition.hxx contains the code used to find partitions among merge lists for each warp to work on.
- src/io-merge.hxx contains methods to merge K lists using the minBlockHeap data structure.
- src/params.h contains definitions of all constants used by the sorting algorithm, including input size, K, number of blocks, etc.
- src/cmp.hxx contains the definition of the comparision function and datatypes used to sort.  The default < comparator to sort integer datatypes is enabled by default.
- scripts/ contains several scripts used to run performance benchmarks of GPU-MMS
