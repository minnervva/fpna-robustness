import torch
from torch.nn import functional as F

def fgsm_attack(model, x, y, epsilon):
    # Make sure x requires gradient
    x.requires_grad = True

    # Forward pass
    logits = model(x)
    
    # Compute the loss
    loss = F.nll_loss(logits, y)
    
    # Zero out previous gradients
    model.zero_grad()
    
    # Compute gradients of the loss w.r.t. the input image
    loss.backward()

    # Create adversarial examples using the sign of the gradients
    x_adv = x + epsilon * x.grad.sign()

    # Ensure the values are still within [0, 1]
    x_adv = torch.clamp(x_adv, 0, 1)
    
    return x_adv