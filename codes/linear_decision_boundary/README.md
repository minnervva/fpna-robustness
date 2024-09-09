# Linear decision boundary example

We present a naive, synthetic example of a Deep Learning (DL) model with a **linear** decision boundary to illustrate the impact of Floating Point Non-Associativity (FPNA) on DL classifiers and regression models. In particular, we consider the model

$$ f : \mathbf{x} \in \mathbb{R}^{d} \longrightarrow \mathbf{1}\{ \hat{\mathbf{n}} \cdot \mathbf{x} \geq b \}$$

where $\hat{\mathbf{n}} \in \mathbf{R}^{d}$ is the normal vector, $b \in \mathbf{R}$ is the bias and $\mathbf{1}(\cdot)$ is the indicator funtion. This binary classifier separates points above and below the hyperplane specified by ${\hat{\mathbf{n}}} \cdot \mathbf{x} = b$ into distint classes. 

Inspired by the classic equation of a straight line $y=x$, we generalise this to arbitraty dimension $d$. We note the equation of the straight line is non-unique, representable as permutations of elements of the normal vector $\frac{1}{\sqrt{d(d-1)}} \cdot (d-1, -1, \dots, -1)$. The equation of the straight line is therfore given as :

$$ \frac{1}{\sqrt{d(d-1)}} \cdot \pi ((d-1, -1, \dots, -1))^{T} \mathbf{x} $$

where $\pi$ is a permutation function.

We begin our analysis by investigating the output for points on the decsion boundary, where $b=0$. We simulate FPNA by iterating over normal vector representations. In this case the theoretical value of the outputs should strictly be zero. However, experiments on an Nvidia H100 produce the output distribution below.

![image](/fpna-robustness/codes/linear_decision_boundary/histogram.png)

We infer a minimum and maximum variation of $\pm 3 \times 10^{-9}$. We can determine a theoretical bound for this variation, given an analysis of floating point numerics <cite>[[1]]</cite>. In particular, given a the IEEE fp32 representation with base $\beta=2$ and precision $p=23$, we can deduce a bound for the maximal relative variation. First, we define the machine precision <cite>[[1]]</cite> , $\epsilon = \frac{\beta}{2}\beta^{-p}$. For the fp32 format, $\epsilon = 2^{-23}$. Given the maximal relative variation for an addition is $2 \epsilon$ and the maximal relative variation for a multiplication is $2 \epsilon$, the relative variation for the equation of a straight line $\hat{\mathbf{n}} \cdot \mathbf{x}$ with bias $b \neq 0$ is 

$$ \| \frac{(\hat{\mathbf{n}} \cdot \mathbf{x} )_{\text{baseline}} - (\hat{\mathbf{n}} \cdot \mathbf{x} )_{\text{baseline}}}{(\hat{\mathbf{n}} \cdot \mathbf{x} )_{\text{baseline}}}\| \leq 3d\epsilon $$ 

TODO @sanjif-shanmugavelu, prove this and tighten bound to remove dependence on dimension, $d$. In the following analysis, we consider points on the decision boundary $\hat{\mathbf{n}} \cdot \mathbf{x} = b$ and iteratively add deviations of $n \cdot \epsilon+ \hat{\mathbf{n}}$ where $\epsilon = 1 \times {10}^{-12}$ and $0 \leq n \leq 3000$. In particular, we note that the percentage of positive predictions decreases with distance from the decision boundary. There is significant variation, which also decreases with distance from the decision boundary. We interpret this curve as points which are $ \geq 3000 \cdot \epsilon$ away from the decision boundary will not have their classification flipped. Alternatively, these points are **robust**.

![image](/fpna-robustness/codes/linear_decision_boundary/adversarial_attack.png)
[1]: https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html
