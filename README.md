# Appendices

This readme provides all necessary details that could not find their place in
the main text due to space restriction. It is divided in two sections. The first
section explains how we instrumentalize the code to retrieve the execution order
of the atomic operations and how to compile and run it. It also contains
additional results for the Mi250x that were only mentioned but not shown in the
main text.

## Measuring the Execution Order of Atomic Operations

In the cuda or HIP programming models, instructions are executed sequentially in
blocks of 32 threads but their scheduling is unknown. It can lead to issues when
floating point atomic operations are used as many of these operations - such as
the four elementary artihmetic operations - are non-associative; a problem also
known as floating point non-assocciativity. Numerical flutuations become more
important and can strongly influence the results in highly non-linear models
such as neural networks. The questions we want to answer are the following.
First can we instrument the code such that we gain insights about the atomic
instructions ordering. Second can we influence instruction scheduling via
external means such as power capping, resource sharing or parallel workloads.
Third the influence of the elements type on the ordering is unexplored here and
merits further studies.

### Instrumentation

To reply to these two questions, we use the reduction algorithm, one of most
used algorithm in computer and numerical science. This algorithm can be
implemented on GPU in two stages, the first stage is a tree reduction ran by
each thread block on a block of data, before storing the partial result in a
temporary array. The second stage is yet again a reduction that is applied to
the temporary array on either CPU or GPU after a global synchronization barrier
is applied. When implemented correctly, this method provides deterministic
results. However, The global synchronization barrier can be avoided
all-to-gether if we use atomic instructions. We can either use an atomic
instruction to update the accumulator; this method does not need extra storage,
or update a counter such that the last block updating the counter is responsible
for calculating the final answer at the cost of extra storage. Both methods are
used in practice but the first method can lead to fluctuating results when
floating point atomic instructions are used while the second method always
provides deterministic results regardless of the instruction scheduling or data
type.

The CUDA programming model does not provide any direct mechanism to follow the
instruction scheduling of atomic operations. However, we can retrieve this
information if we can register the value of the accumulator or the counter as
function of the block index before upating one of the two variables with an
atomic instruction. Doing this 

```[c++] 
   track[blockIdx.x] = acc; 
   atomicAdd(&acc, partial_sum);
``` 

does not work, because the CUDA programming model does not garranty that the
value of the `acc` variable is valid as it can also be updated by another thread
block at the same time. We can use the return value instead as artihmetic atomic
instructions always return the value of the updated variable before updating it.
More specifically,

```[c++]
   track[blockIdx.x] = atomicAdd(&acc, partial_sum); 
```

where the `track` array is used to store the value of the accumulator vs the
block index. The full code is given by
```[c++]
template <class T, unsigned int blockSize> 
__global__ void reduce_gpu_follow_atomic(const T *g_idata, T *g_odata, unsigned int n) {
    // Handle to thread block group
  T *sdata = SharedMemory<T>();

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
  unsigned int gridSize = blockSize * 2 * gridDim.x;
  // use the end of scratch memory to store how many block are already done.

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

  blockReduce<T, blockSize, warp_reduction_method::shared_memory>(sdata, mySum, tid);

  if (tid == 0) {
    T temp = atomicAdd(&g_odata[0], mySum);
    g_odata[blockIdx.x + 1] = temp;
  }
}
```

The code located in the directory `codes/instruction_ordering` provides both
implementations but we use the implementation above in the main text. Choosing
$x_i \ neq x_j, x_i > 0$, imply $\sum_{j < i} x_j < \sum_{j < i + 1} x_j$. We
retrieve the instruction scheduling after sorting the variable `g_odata`. 

It is also possible to use the counter value to retrieve the isntruction scheduling.
The GPU kernel can also be found in the code.
 
### Compiling and Running The Code

The code measuring the instruction ordering can be found in the directory
`codes/instruction_ordering`. It supports both `cuda` and `hip` and its only
dependencies are a `c++` compiler (gcc > 11), CUDA/HIP and recent version of
`cmake`. All the other dependencies will be installed automatically if
necessary. To compile the code, go to `codes/instruction_ordering` and run the
following commands

```[bash]
cd codes/instruction_ordering
mkdir build
cd build
cmake -DREDUCE_USE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80
```
for the Nvidia GPU (A100) or 
```
cd codes/instruction_ordering
mkdir build
cd build
cmake -DREDUCE_USE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a
```

for AMD GPU (Mi250x in that specific example). The GPU architecture can be
controlled by the variables `CMAKE_CUDA_ARCHITECTURES` or
`CMAKE_HIP_ARCHITECTURES` depending on the targeted platform. Typing `make` in
the command line will compile two programs, `test_reduce` which measures the
instruction ordering under various conditions and `KendallTau` which calculates
the Kendall tau correlation function between two matrices stored in `csv` files.
Use the argument `--help` to get all program arguments.

We use the following command to run the application.

```
./test_reduce --mapping_under_load --mapping --number_of_samples 1000 --max_reduction_size 1000000 --matrix_size 4096
```

The program first calculates the instruction ordering in absence of parallel
workload then repeats the same measurements while running a matrix-matrix
product in parallel on the GPU. The GPU can be selected with the environment
variable `CUDA_VISIBLE_DEVICES` or `HIP_VISIBLE_DEVICES` on a multi-gpu machine
followed by either an integer or id string
`MIG-8ffb15ae-81b3-5204-a73d-1668580a36b6` (for GH200 MiG configurations).

The results are stored in several `csv` files whose name encode the `GPU`
architecture and the workload type. File with name
`mapping_atomicInc_block_index_GH200.csv`
(`atomic_ops_ordering_under_load_GH200.csv`)) for instance states that we
measure the instruction order in absence (presence) of external workload on a
GH200 GPU. This file contains a list of already sorted block index versus
execution order. All these numbers are integer.

The second application calculates the Kendall Tau correlation function on GPU.
Please refers to the
[wikipedia](https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient)
page and references therein for a full description of its properties. Its
implementation uses atomic instructions on integers which are not sensitive to
instruction order, making the result platform independent. The application
compares two csv files generated by the `test_reduce` application and returns
the lower triangular part of the correlation matrix. To get the correlation
matrix, simply enter

```
./KendallTau -N 1000 -n 7913 mapping_atomicInc_block_index_GH200.csv atomic_ops_ordering_under_load_GH200.csv -o my_file.csv
```

to compare two different files or

```
./KendallTau -N 1000 -n 7913 mapping_atomicInc_block_index_GH200.csv -o my_file.csv
```

to calculate the correlation within the same file. The following syntax is also
allowed.


```
./KendallTau -N 1000 -n 7913 mapping_atomicInc_block_index_GH200.csv mapping_atomicInc_block_index_GH200.csv -o my_file.csv
```

The second application returns a `csv` file full of floating point numbers
representing the distribution of the kendall tau correlation. Plotting the
histograms of this distribution is done externally. A mathematica script is
provided for convenience.

### Results



# Robustness of Deep Leaning Applications to Floating Point Non-Associativity

Previous work investigated the impact of Floating Point Non-Associativity (FPNA) <cite>[1]</cite> on simple isolated operations and an end-to-end training and inference loop. The focus was a Graph Neural Network (GNN), and the offending non-deterministic operation was `index_add`. We used variability metrics to quantify the impact of FPNA. However, we did not explore the impact of FPNA on application specific workloads, leaving this for future work. In this paper, we investigate the impact of FPNA on the robustness of general DL classifiers.

We noticed FPNA being responsible for flipping the output of a  multiclass classifier. This demonstrates FPNA as a non algorithmic, non explicitly crafted “attack”. Robustness against FPNA must be iinvestigated in a similar vein to all other attacks, ensuring the robustness, reliability and replicability of ML models across hardware and software platforms.


## Paper and Repository Organisation
We organise our findings as follows:

- Introduction
- Metrics and Methodology
- Simple synthetic example of a [linear decision boundary](/codes/linear_decision_boundary) to illustrate the problem  - @sanjif-shanmugavelu
- Experiments with synthetic non-determinism on linear layers to illustrate the impact of non-determinism on DL training and inference workloads, as a function of FPNA severity. @chrisculver
- Experiments similar to before, with real world DL models, specifically [GNNs](/codes/linear_decision_boundsry)
- Future Work and Conclusion

TODO: add project directory structure after discussing with @chrisculver


## Metrics
To quantify the bitwise non-determinism between two implementations of a function producing a multidimensional array output, we define two different metrics. Let two implementations of a function $f$ be $f_1$ and $f_2$, which produce as outputs the arrays $\mathbf{A}$ and $\mathbf{B}$, respectively, each with dimensions $d_1, d_2, \ldots, d_k$ and $D$ total elements. The first metric is the elementwise relative mean absolute variation,

$$V_{\text{ermv}}(f) = \frac{1}{D} \sum_{i_1=1}^{d_1} \cdots \sum_{i_k=1}^{d_k} \frac{|A_{i_1, \ldots, i_k} - B_{i_1, \ldots, i_k}|}{|A_{i_1, \ldots, i_k}|}.
$$

The second metric, the count variability, measures how many elements in the multidimensional array are different, 

$$
V_c(f) = \frac{1}{D} 
\sum_{i_1=1}^{d_1} \cdots \sum_{i_k=1}^{d_k} \mathbf{1}(A_{i_1, i_2, \ldots, i_k} \ne B_{i_1, i_2, \ldots, i_k})
$$

where $\mathbf{1}(\cdot)$ is the indicator function, which is 1 if the condition inside the parentheses is true, and 0 otherwise. 

Each of these metrics is zero if and only if the two multidimensional arrays $\mathbf{A}$ and $\mathbf{B}$ are bitwise identical. The count variability $V_c$ informs us of the percentage of varying array elements between the two output arrays, while the $V_{\text{ermv}}$ produces a global metric for the variability of the array outputs.

## Introduction


Let $f : \mathbb{R}^{d} \longrightarrow \mathbb{R}^{L}$, where $d, L \in \mathbb{N}$ be an arbitrary multiclass classifier. Given a datapoint $\mathbf{x} \in \mathbb{R}^{d}$, the class which the classifier $f$ predicts for $\mathbf{x}$ is given by $\hat{k}(\mathbf{x}) = \operatorname{\argmax}_{k} f_{k}(\mathbf{x})$, where $f_{k}(\mathbf{x})$ is the $k$-th component of $f(\mathbf{x})$.
Note $\operatorname{\argmax}_{k} \hat{k}(\mathbf{x})$ may not be unique. In this case, without loss of generality, we take $\hat{k}(\mathbf{x})$ to be the firstc component where $f$ achieves its maximum.

The confidence of classification $F$ at point $\mathbf{x} \in \mathbf{R}^{d}$ is given by

$$F(\mathbf{x}) = f_{\hat{k}(\mathbf{x})} - \operatorname{max}_{k \neq \hat{k}(\mathbf{x})} f_{k}({\mathbf{x}})$$

$F$ describes the difference between the likelihood of classification for the most probable class and the second most probable class. Note like before this second most probable class need not be unique. 
For a given $\mathbf{x} \in \mathbf{R}^{d}$, the larger the value of $F(\mathbf{x})$, the more confident we are in the prediction $\hat{k}(\mathbf{x}) = \operatorname{\argmax}_{k} f_{k}(\mathbf{x})$ given by the classifier.

The decision boundary $B$ of a classifier $f$ is defined as the set of points $f$ is equally likely to classify into at least two *distinct* classes.

$$ B = \{\mathbf{x} \in \mathbb{R}^{d} :  F(\mathbf{x}) = 0\} $$

$B$ splits the domain, $\mathbb{R}^{d}$ into subspaces of similar classification, $C_{k} = \{ \mathbf{x} \in \mathbb{R}^{d} : \hat{k}(\mathbf{x}) =k, 1 \leq k \leq L\}$ Therefore, assuming each $C_{k}$ has non-empty interior, $\mathbf{R}^{d} \setminus B = \bigcup_{k=1}^{L} C_{k}$. In particular, $\mathbf{x} \notin B$ implies $x \in C_{k}$. Given $x \in \mathbb{R}^{d}$, a perturbation $\delta({\mathbf{x}}) \in \mathbb{R}^{d}$ such that $\mathbf{x} + \delta({\mathbf{x}}) \in B$, we have that $\mathbf{x} + \delta({\mathbf{x}})$ is on the boundary of misclassification. Hence, when considering misclassification, we study properties of the decision boundary $B$.

An adversarial perturbation, $\boldsymbol{\delta}_{\text{adv}}(\mathbf{x}; f)$ is defined by the following optimisation problem:

$$
\boldsymbol{\delta}_{\text{adv}}(\mathbf{x} ; f)= \operatorname{\argmin}_{\boldsymbol{\delta} \in \mathbb{R}^d}\|\boldsymbol{\delta}\| \text {s.t.} F(\mathbf{x}+ \boldsymbol{\delta}) = 0
$$

In other words, $\boldsymbol{\delta}_{\text{adv}}(\mathbf{x} ; f)$ is a perturbation vector of minimal length to the decision boundary. Note $\boldsymbol{\delta}_{\text{adv}}(\mathbf{x} ; f)$ need not be unique. In this case, without loss of generality, we select a valid perturbation vector at random and assign it to $\boldsymbol{\delta}_\text{adv}(\mathbf{x} ; f)$. Point $\mathbf{x}+\boldsymbol{\delta}_{\text{adv}}(\mathbf{x} ; f)$ will then be on the boundary of misclassification. Note an adversarial perturbation is typically defined as the minimal distance to mislcassification, $\min _{\boldsymbol{\delta} \in \mathbb{R}^d}\|\boldsymbol{\delta}\|$ s.t. $\hat{k}(\mathbf{x}+\boldsymbol{\delta}) \neq \hat{k}(\mathbf{x})$. However, it is easier to consider a perturbation to the decision boundary and we shall consider these two frameworks as identical throughout this paper. This is because once on the decision boundary, an infinitesimal perturbation will result in misclassification.

An adversarial distance, $\delta_{\text{adv}}(\mathbf{x} ; f)$ is defined as:
$$
\delta_{\text{adv}}(\mathbf{x} ; f)=\|\boldsymbol{\delta}_{\text{adv}}(\mathbf{x} ; f)\|=\operatorname{\min} _{\boldsymbol{\delta} \in \mathbb{R}^d}\|\boldsymbol{\delta}\| : F(\mathbf{x}+\boldsymbol{\delta})=0
$$

In other words, $\delta_{\text{adv}}(\mathbf{x} ; f)$ is the minimal length of a perturbation to the decision boundary. Given $\alpha \in \mathbb{R}$, if $\alpha \geq \delta_{\text{adv}}(\mathbf{x} ; f)$, then $\exists \boldsymbol{\delta} \in \mathbb{S}=\{\mathbf{x} \in \mathbb{R}^d : \|\mathbf{x}\|=1\}$ such that $F(\mathbf{x}+\alpha \boldsymbol{\delta})=$ 0 . On the other hand, if $\alpha<\delta_{\text{adv}}(\mathbf{x} ; f), \forall \boldsymbol{\delta} \in \mathbb{S}$ we have that $F(\mathbf{x}+\alpha \boldsymbol{\delta}) \neq 0$. All perturbations of magnitude less than $\alpha$ cannot reach the decision boundary $B$ . The larger the value of $\delta_{\text{adv}}(\mathbf{x} ; f)$, the more robust we say the point $\mathbf{x} \in \mathbb{R}^d$ is to adversarial perturbations.


An adversarial attack is a minimal perturbation. For a random perturbation, given a rate of misclassification $\epsilon \in [0, 1]$,  and a measure $\mu$, $\Delta_{\mu, \varepsilon}(\mathbf{x} ; f)$ denotes the maximal radius of the sphere centred at $\mathbf{x}\in\mathbb{R}^{d}$ such that the probability of misclassification of datapoints on $\mathbf{x} + \delta\mathbb{S}$ is less than or equal to $\epsilon$. Note that $\mathbb{S}$ denotes the unit sphere centred at the origin in $\mathbb{R}^{d}$. 

$$\Delta_{\mu, \varepsilon}(\mathbf{x} ; f)=\max_{\boldsymbol{\delta} \in \delta\mathbb{S}} \|\boldsymbol{\delta}\| \ : \ \mathbb{P}_{\delta\mathbb{S}} \left(\arg \max _{k} f_{k}(\mathbf{x})=\hat{k}(\mathbf{x}) \neq \hat{k}(\mathbf{x}+\boldsymbol{\delta})\right) \leqslant \varepsilon$$ 

## Questions we hope to answer

- Robustness bounds and results have been studied extensively in the case of inpt variability. These bounds do not include the effects of to run-by-run FPNA variability. Can we replicate and compare well known robustness results with deterministic algorithms and non-deterministic algorithms? What is the extent of the divergence, if any? Loop over hardware, ask @asedova and @mtaillefumier for help running experiments.
- Can we extend this to the regression case? Ask @asedova for papers.
- Does adversarial training help with FPNA attacks? Does adversarial training on machine $A$ translate to similar robustness on machine $B$ when $A \neq B$.
- Random perturbation training has shown to improve robustness, can training with non-deterministic algorithms help in a similar vein? Compare the two forms of adversarial training. 



[1] : [Impacts of floating-point non-associativity on reproducibility for HPC and deep learning applications](https://arxiv.org/abs/2408.05148)
