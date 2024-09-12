# Adversarial and Random Attacks

Pertubations to input data via adversarial and random attacks are crafted to mislead machine learning models by making small, imperceptible changes to the input data.

## Fast Gradient Sign Method (FGSM)

FGSM perturbs input data in the direction of the gradient of the loss function with respect to the input data. We maximize the loss function which can lead to a significant decrease in the model’s accuracy.


Given a model $f$ with parameters $\theta$, an input image $\mathbf{x}$, and its corresponding label $y$, the FGSM generates an adversarial example $\mathbf{x}^{'}$ by perturbing $\mathbf{x}$ scaled by $\epsilon$ as follows:

$$
\mathbf{x}^{'} = \mathbf{x} + \epsilon \cdot \text{sign} ( \nabla_{\mathbf{x}} J(f(\mathbf{x}; \theta), y)$$
