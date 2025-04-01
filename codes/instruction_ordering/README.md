# how to compile the code and run it

The only dependencies of this code are CUDA ro HIP, a C++ compiler starting from gcc 11 and cmake 3.24.

To compile it, clone the repository and `cd` into the src directory
```bash
mkdir build
cd build
cmake -DCMAKE_CUDA_ARCHITECTURES=80 -DREDUCTION_USE_CUDA=ON ..
make
```

the binary is called `test_reduce`. To get the mapping without parallel workload 
```bash
./test_reduce --mapping
```
and with workload
```bash
../../../build//test_reduce --mapping_under_load --number_of_samples 100 --max_reduction_size 1000000 
```
It will generate two csv files with the block index vs execution order. It is
possible to change the workload by changing the size of the matrices with
`--matrix_size 2048` whatever value is needed.

**WARNING** if run multiple times over the files will be **overwritten**. To
change this behavior add `--append` to the program options.

DO NOT expect reproducible results when run on the same GPU model but different
nodes or computers. Even compiling the same source code might give different
answers :(.

If one wants to run the code multiple times then these few lines of bash should
do the trick

```bash
for (i=0;i<10;i++); do ../../../build//test_reduce --mapping_under_load --number_of_samples 100 --max_reduction_size 1000000 --mapping --append; sleep 2; done
```

# Possible vectors of fpna attack 

FPNA attacks are solely due to the unknown execution order of floating point
atomic operations. To make matter more complicated, the official CUDA
documentation contains very little information about the actual hardware
implementation of the atomic operations and only very few reverse ingeneering
studies are publicly available. The main argument about fpna attack is that some
permutations have a bigger impact on classification than others but is it
possible to actually generate them using for instance external loads or other
physical constraints such as temperature. 

Atomic operations by their nature have at least three potential sources of
non-determinism, (i) the block scheduling which associates a given block to a
given symmetric mulprocessing unit, (ii) the instruction sheduler and (iii) the
dedicated execution unit that executes the atomic instructions. All three units
operate within resources constraints that make any analysis of kernels with
floating atomic operations difficult at the numerical level. 

Besides resources, GPU also have protection mechanisms to avoid overheating for
instance. They can also run other kernels which on their own will limit
resources available to the kernels using atomic instructions. All these
constraints can impact the execution order of the atomic instrictions and in
turn generate a different answer with possible nefast consequences. Showing that
it is possible to impact the execution order of atomic instructions is the
subject of this entire section.

```c++
template <class T, unsigned int blockSize> 
__global__ void reduce_gpu_follow_atomic(const T *g_idata, T *g_odata, unsigned int n) {
  extern __shared__ int *idata[];
  T *sdata = (T *)idata;

  unsigned int tid = threadIdx.x;

  if (tid + blockIdx.x * blockDim.x >= n)
    sdata[tid] = 0;
  else
    sdata[tid] = g_idata[tid + blockIdx.x * blockDim.x]
  
  __syncthreads();
  
  for (int offset = 64 / 2; offset > 0; offset /= 2)
    if (tid < offset)
        sdata[tid] += sdata[tid + offset];
    __syncthreads();

  if (tid == 0) {
    T temp = atomicAdd(&g_odata[0], mySum);
    // recording the value of the accumulator before updating it. 
    // The first element is used to store the final sum
    g_odata[blockIdx.x + 1] = temp;
  }
}
```

# Methodology

We use the parallel sum algorithm to show that it is indeed possible to modify
the execution order of atomic instructions. The parallel sum or the reduce
function is defined as $\sum_i^n x_i$ where $x_i$ is a set of \gls{FP64}
numbers. This operation is simple to implement on GPU (see
listing.\ref{listing:sum:1}) especially with the atomicAdd operation. This
implementation does not give us direct access to the order in which the
atomicAdd instruction is executed but it is still possible to get the ordering
{\it a forciori} if $x_i>0$. In that case, The value of the accumulator grows
monotonically every time a thread block updates it with the atomicAdd
instruction. The atomicAdd instruction also returns the value of the accumulator
before the update which can be recorded in a table indexed with the block index.
The order in which the atomic instructions are executed can be obtained after
applying a sort by index on the table containing the partial accumulation
results.

We generate 100 lists of of floating point numbers taken from the uniform
distribution $U(0, 1)$ and calculate the sum of these 100 arrays 10 times with
and without additional workload running in parallel. The additional workload
contains the same number of steps as the reduction steps but runs asynchronously
compared to the sum workload. This can be achieved by running the two different
workloads in two different cuda or hip streams. We impose no synchronization
between the two workloads either. Each individual step of the matrix-matrix
multiplication workload is composed of four sub-steps. First we generate three
matrices of fixed size filled with random number taken from the uniform
distribution $U(0, 1)$ and then use cublas/hipblas to calculate the
matrix-matrix multiplication.

We repeat the same methodology varying the matrices sizes, power limitations and
GPU partitioning when possible. 



the code and scripts are provided in the github repository associated to this
paper.
# operations 

We know that some specially crafted permutations can change the result of a
neural network during inference. Right now, this permutation is hand crafted
(well we search for such permuation) and targeting specific opeartors used in ML
workload. The problem is then can we induce these permutations externally,
through for instance temperature or frequency switching. Note that temperature
and throttling are already known vectors of attack to extract information from a
GPU or a CPU. So the probability that we can switch a classifier response with
just temperature or throttling is actually high. We just need to show it (which
is the hard part)

They are several target function where this might happen. the `scatter_reduce`
function pytorch for instance is fully custom and does not seem to rely on any
external library which means that we would have to modify pytorch itself to
monitor the non-deterministic behavior. 

The scatter_reduce function is called `reduce_by_key` in cub / hibcub and it is
**not** deterministic (it was before not anymore). so we can really look at this
implementation either (although the code is available). 

# The reduce function

This operator is used everywhere, (blas, etc) and is a good candidate to study
the ordering of atomic ops on GPU. We already have a deterministic
implementation of it (the cub implementation is also deterministic) and we can
monitor when the atomic ops happens and what block scheduled it. 

## Measuring the order of atomic operations

There is no direct way to get direct access to the scheduling of atomic
operations in general and atomic FP operations in particular. We can however
measure it undirectly for the reduce function if we tailor the initial array
such that the value of the accumulator increases when each thread block calls
the atomicAdd instruction. It can be achieved with any positive random
distributions such as the uniform distribution $U(0,1)$. The atomic instructions
always return the previous values of the variable to be updated. We can use this
at our advantage and store the value in a temporary array indexed by the block
Id. These two lines of codes illustrate the principle

```c++
// do the tree reduction as usual

T prev_sum = atomicAdd(&accumulator, mysum)
partial_sum_[blockIdx.x] = prev_sum;
```

There is no direct monitoring only a global memory write which means that the
operation itself does not or has minimal impact on the instruction scheduling. 
More importantly it happens after the `atomicAdd`

To get the order of the atomic operations, we simply compute the permutation
that sorts the array `partial_sum_` in ascending order. Doing this allows us to
map the atomicAdd ordering to the value of the scalar variability. 



## Atomic operation order old method

We monitor the executation order of the atomic operation during the reduce
kernel execution by exploiting the counter that is incremented atomically by 1
for each block. The last block is chosen such that the counter reach the block
grid size - 1. We can do this without changing the code extensively the only
price to pay is a global memory update. The full code is given by

```c++
template <class T, unsigned int blockSize>
__global__ void
reduce_register_atomic_gpu(const T *__restrict__ g_idata, T *g_odata,
                           int *__restrict__ block_index__, unsigned int n) {
  // Handle to thread block group
  T *sdata = SharedMemory<T>();
  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
  unsigned int gridSize = blockSize * 2 * gridDim.x;
  // use the end of scratch memory to store how many block are already done.

  unsigned int *retirementCount = (unsigned int *)(g_odata + gridDim.x + 1);
  T mySum = 0;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridDim).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  while (i < n) {
    mySum += g_idata[i];

    // ensure we don't read out of bounds -- this is optimized away for powerOf2
    // sized arrays
    if (i + blockSize < n)
      mySum += g_idata[i + blockSize];

    i += gridSize;
  }

  blockReduce<T, blockSize, warp_reduction_method::shared_memory>(sdata, mySum,
                                                                  tid);
  // write result for this block to global mem. We do not optimize away the
  // global write when we use the single pass algorithm with determinism
  // because it would otherwise make the code non deterministic which is
  // something we want to avoid at all costs.
  if (tid == 0)
    g_odata[blockIdx.x] = mySum;
  // We have the option to let the last block do the final reduction instead
  // of either calling the function once more or copy the partial results to
  // CPU and do the final reduction on CPU. Both ways are deterministic but
  // will lead to slightly different answers because of the order of the
  // operations.

  if (gridDim.x > 1) {
    __shared__ bool amLast;

    // wait until all outstanding memory instructions in this thread are
    // Finished
    __threadfence();

    // Thread 0 takes a ticket
    if (tid == 0) {
      unsigned int ticket = atomicInc(retirementCount, gridDim.x);
      
      // we record the block index reaching this point using the ticket as an index.
      // the ticket is effectively the time in which the atomic instruction is executed and returns.
      // We do not modify anything else and monitoring only imply a memory update.
      block_index__[ticket] = blockIdx.x;
      // If the ticket ID is equal to the number of blocks, we are the last
      // block!
      amLast = (ticket == gridDim.x - 1);
    }

    __syncthreads();

    // The last block sums the results of all other blocks
    if (amLast) {
      int i = tid;
      mySum = (T)0;

      while (i < gridDim.x) {
        mySum += g_odata[i];
        i += blockSize;
      }

      blockReduce<T, blockSize, warp_reduction_method::shared_memory>(
          sdata, mySum, tid);

      if (tid == 0) {
        g_odata[0] = mySum;
      }
    }
  }
}
```

Note that this kernel returns a deterministic result which means that we follow
the `atomicInc` instruction execution not the `atomicAdd`. However if we asuume
that only one unit executes these instruction it should still give us an
idea about the execution of all atomic operations independently of the
variable type. Following the execution of the `atomicAdd` will require more
thinking as we can not really do this

```c++
atomicInd(counter, gridSize);
atomicAdd(sum, &partial_sum);
```

because the first call is blocking and we do not know then the second
instruction is executed. We know however when it is scheduled. 
<p align="center">
<img src="figures/Histogram_mapping_atomic_Initial_phase_V100.png" width=300"> 
<img src="figures/Histogram_mapping_atomic_equilibrium_V100.png" width=300"
title="PDF of the difference between the block index and execution time on the V100">
</p>
<p align="center">
<img src="figures/Histogram_mapping_atomic_Initial_phase_Mi250X.png" width=300">
<img src="figures/Histogram_mapping_atomic_equilibrium_Mi250X.png" width=300"
title="PDF of the difference between the block index and execution time on the Mi250X">
</p>
<p align="center">
<img src="figures/Histogram_mapping_atomic_Initial_phase_GH200.png" width=300">
<img src="figures/Histogram_mapping_atomic_equilibrium_GH200.png" width=300"
title="PDF of the difference between the block index and execution time on the GH200">
</p>

these figures show interesting patterns and information about the scheduling.
The ordering is not completely random as the pdf has a bell shape which means
there is some correlations. 
