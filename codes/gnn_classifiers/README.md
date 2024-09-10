# Graph Neural Network (GNN) example

Inspired by previous work investigating the impact of FPNA on Graph Neural Network output variability, we study both the posibility and extent of FPNA inducing flipped output classifications on GNNs <cite>[1]</cite>. We note operations which are key features of GNNs, including `indes_add` and `scatter_reduce` exhibit significant non-determinism <cite>[1]</cite>. We focus on classic GNN benchmark datasets, including CORA and PubMed. To aid visulisation and understanding, we also include image datasets such as MNIST, Fashion MNIST, Street View House Numbers (SVHN) and
CIFAR-10. We leverage superpixel representation of these images via the Simple Linear Iterative Clustering (SLIC) algorithm and process them into graphs with the Region Adjacency Graph (RAG) algorithm <cite>[2]</cite>.

Here is a gif with the 2D grayscale image representation (left) and graph represenation (right) of the classic MNIST dataset.  

![MNIST Image and Graph Representation GIF](/fpna-robustness/codes/gnn_classifiers/MNIST_superpixels.gif)

[1] : [Impacts of floating-point non-associativity on reproducibility for HPC and deep learning applications](https://arxiv.org/abs/2408.05148)


[2] : [Superpixel Image Classification with Graph Attention Networks](https://arxiv.org/abs/2002.05544)