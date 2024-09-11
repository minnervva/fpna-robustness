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

$$ B = \{\mathbf{x} \in \mathbb{R}^{d} :  F(\mathbb{x} = 0)\} $$

$B$ splits the domain, $\mathbb{R}^{d}$ into subspaces of similar classification, $C_{k} = \{ \mathbf{x} \in \mathbb{R}^{d} : \hat{k}(\mathbf{x} =k), 1 \leq k \leq L\}$ Therefore, assuming each $C_{k}$ has non-empty interior, $\mathbf{R}^{d} \setminus B = \bigcup_{k=1}^{L} C_{k}$. In particular, $\mathbf{x} \notin B$ implies $x \in C_{k}$. Given $x \in \mathbb{R}^{d}$, a perturbation $\delta({\mathbf{x}}) \in \mathbb{R}^{d}$ such that $\mathbf{x} + \delta({\mathbf{x}}) \in B$, we have that $\mathbf{x} + \delta({\mathbf{x}})$ is on the boundary of misclassification. Hence, when considering misclassification, we study properties of the decision boundary $B$.

An adversarial perturbation, $\delta_{\text{adv}}(\mathbf{x} ; f)$ is defined by the following optimisation problem:
$$
\delta_{\text{adv}}(\mathbf{x} ; f)=\underset{\delta \in \mathbb{R}^d}{\arg \min }\|\delta\|_2 \text { s.t } F(\mathbf{x}+\delta)=0
$$

In other words, $\delta_{\text{adv}}(\mathbf{x} ; f)$ is a perturbation vector of minimal length to the decision boundary. Note $\delta_{\text{adv}}(\mathbf{x} ; f)$ need not be unique. In this case, without loss of generality, we select a valid perturbation vector at random and assign it to $\delta_{a d v}(\mathbf{x} ; f)$. Point $\mathbf{x}+\delta_{a d v}(\mathbf{x} ; f)$ will then be on the boundary of misclassification. Note an adversarial perturbation is typically defined as the minimal distance to mislcassification, $\min _{\delta \in \mathbb{R}^d}\|\delta\|_2$ s.t $\hat{k}(\mathbf{x}+\boldsymbol{\delta}) \neq \hat{k}(\mathbf{x})$. However, it is easier to consider a perturbation to the decision boundary and we shall consider these two frameworks as identical throughout this paper. This is because once on the decision boundary, an infinitesimal perturbation will result in misclassification.

An adversarial distance, $\delta_{\text {adv }}(\mathbf{x} ; f)$ is defined as:
$$
\delta_{a d v}(\mathbf{x} ; f)=\left\|\delta_{a d v}(\mathbf{x} ; f)\right\|=\min _{\delta \in \mathbb{R}^d}\|\delta\|_2 : F(\mathbf{x}+\delta)=0
$$

In other words, $\delta_{\text{adv}}(\mathbf{x} ; f)$ is the minimal length of a perturbation to the decision boundary. Given $\alpha \in \mathbb{R}$, if $\alpha \geq \delta_{\text {adv }}(\mathbf{x} ; f)$, then $\exists \delta \in \mathbb{S}=\left\{\mathbf{x} \in \mathbb{R}^d \mid\|\mathbf{x}\|=1\right\}$ such that $F(\mathbf{x}+\alpha \boldsymbol{\delta})=$ 0 . On the other hand, if $\alpha<\delta_{a d v}(\mathbf{x} ; f), \forall \delta \in \mathbb{S}$ we have that $F(\mathbf{x}+\alpha \delta) \neq 0$. All perturbations of magnitude less than $\alpha$ cannot reach the decision boundary B . The larger the value of $\delta_{\text{adv}}(\mathbf{x} ; f)$, the more robust we say the point $\mathbf{x} \in \mathbb{R}^d$ is to adversarial perturbations.



[1] : [Impacts of floating-point non-associativity on reproducibility for HPC and deep learning applications](https://arxiv.org/abs/2408.05148)
