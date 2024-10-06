import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import warnings


def assign_fixed_params(model):
    rng_gen = torch.Generator()
    rng_gen.manual_seed(123)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(*p.shape, generator=rng_gen, dtype=p.dtype))


class AtomicLinear(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(AtomicLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.randn(out_features))

    # x has dimension [batch_size, in_features]
    def forward(self, x):
        # return x.matmul(self.weight.t()) + self.bias
        # percent_flipped = 0.5
        # [0,1,2,3] + torch.randperm([4,5,6,7])
        y = x[:, None, :] * self.weight
        # with torch.no_grad():
        indices = torch.randperm(self.in_features)
        # print(indices)
        prod = y[:, :, indices]
        return prod.sum(dim=2) + self.bias[None, :]


class MNISTBase(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        batch_size, channels, width, height = x.size()

        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)

        # layer 1 (b, 1*28*28) -> (b, 128)
        x = self.layer_1(x)
        x = torch.relu(x)

        # layer 2 (b, 128) -> (b, 256)
        x = self.layer_2(x)
        x = torch.relu(x)

        # layer 3 (b, 256) -> (b, 10)
        x = self.layer_3(x)

        # probability distribution over labels
        x = torch.log_softmax(x, dim=1)

        return x


class MNISTClassifier(MNISTBase):
    def __init__(self):
        super().__init__()
        # mnist images are (1, 28, 28) (channels, width, height)
        self.layer_1 = torch.nn.Linear(28 * 28, 128)
        self.layer_2 = torch.nn.Linear(128, 256)
        self.layer_3 = torch.nn.Linear(256, 10)
        assign_fixed_params(self)


class AtomicMNISTClassifier(MNISTBase):
    def __init__(self):
        super().__init__()
        # mnist images are (1, 28, 28) (channels, width, height)
        self.layer_1 = AtomicLinear(28 * 28, 128)
        self.layer_2 = AtomicLinear(128, 256)
        self.layer_3 = AtomicLinear(256, 10)
        assign_fixed_params(self)


class AstroBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.nfeatures = 8
        self.nhidden = 5 * self.nfeatures
        self.nclasses = 3

        self.norm1 = torch.nn.BatchNorm1d(self.nhidden)
        self.norm2 = torch.nn.BatchNorm1d(self.nhidden)
        self.activation = torch.nn.ReLU()
        self.probs = torch.nn.Softmax()

    # x (b, 8)
    def forward(self, x):
        y0 = self.norm1(self.activation(self.layer0(x)))
        y1 = self.norm2(self.activation(self.layer1(y0)))
        y2 = self.layer2(y1)
        return y2


class AstroClassifier(AstroBase):
    def __init__(self):
        super().__init__()
        self.layer0 = torch.nn.Linear(self.nfeatures, self.nhidden)
        self.layer1 = torch.nn.Linear(self.nhidden, self.nhidden)
        self.layer2 = torch.nn.Linear(self.nhidden, self.nclasses)
        assign_fixed_params(self)


class AtomicAstroClassifier(AstroBase):
    def __init__(self):
        super().__init__()
        self.layer0 = AtomicLinear(self.nfeatures, self.nhidden)
        self.layer1 = AtomicLinear(self.nhidden, self.nhidden)
        self.layer2 = AtomicLinear(self.nhidden, self.nclasses)
        assign_fixed_params(self)

class LearnablePermutation(torch.nn.Module):
    def __init__(self, in_features, temperature=1.0):
        super(LearnablePermutation, self).__init__()
        self.in_features = in_features
        self.temperature = temperature
        self.logits = torch.nn.Parameter(torch.randn(in_features, in_features))
        self.init_identity()

    def init_identity(self):
        with torch.no_grad():
            self.logits.data = torch.eye(self.in_features)
    
    def forward(self):
        if self.training:
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(self.logits)))
            perm_matrix = F.softmax((self.logits + gumbel_noise) / self.temperature, dim=-1)
        else:
            perm_matrix = torch.zeros_like(self.logits)
            row_indices, col_indices = linear_sum_assignment(-1 * self.logits.detach().cpu().numpy(), maximize=True)
            perm_matrix[row_indices, col_indices] = 1.0
            
        self._validate_permutation_matrix(perm_matrix) 
        return perm_matrix
    
    def _validate_permutation_matrix(self, perm_matrix):
        if self.training:
            row_sums = perm_matrix.sum(dim=-1)
            if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5):
                warnings.warn("Warning: In training mode, some rows of the permutation matrix do not sum to 1")
        else:
            if not torch.all((perm_matrix == 0) | (perm_matrix == 1)):
                warnings.warn("Warning: In eval mode, matrix contains elements that are not 0 or 1")
            
            if not (torch.all(perm_matrix.sum(dim=-1) == 1) and torch.all(perm_matrix.sum(dim=-2) == 1)):
                warnings.warn("Warning: In eval mode, not all rows and columns have exactly one '1'")
  
    
class AtomicLinearLearnablePermutation(torch.nn.Module):
    def __init__(self, in_features, out_features, temperature=1.0):
        super(AtomicLinearLearnablePermutation, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.randn(out_features))
        self.permutation = LearnablePermutation(in_features, temperature)

    def forward(self, x):
        y = x[:, None, :] * self.weight
        perm_matrix = self.permutation()
        permuted_y = torch.matmul(y, perm_matrix)
        return permuted_y.sum(dim=2) + self.bias[None, :]
    

class MNISTClassifierAtomicLinearLearnablePermutation(torch.nn.Module):
    def __init__(self, temperature=1.0):
        super(MNISTClassifierAtomicLinearLearnablePermutation, self).__init__()
        self.fc1 = AtomicLinearLearnablePermutation(28 * 28, 256, temperature)
        self.fc2 = AtomicLinearLearnablePermutation(256, 512, temperature)
        self.fc3 = AtomicLinearLearnablePermutation(512, 1024, temperature)
        self.fc4 = AtomicLinearLearnablePermutation(1024, 512, temperature)
        self.fc5 = AtomicLinearLearnablePermutation(512, 256, temperature)
        self.fc6 = AtomicLinearLearnablePermutation(256, 128, temperature)       
        self.fc7 = AtomicLinearLearnablePermutation(128, 10, temperature)                                          

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = self.fc7(x) 
        return x
    
    
def freeze_permutations(model: nn.Module, freeze_permutation: bool=True) -> None:
    """ Freeze or unfreeze learnable permutation and other sub module ```requires_grad``` flag

    Args:
        model (nn.Module): _description_
        freeze_permutation (bool): _description_
    """
    for module in model.modules():
        if isinstance(module, LearnablePermutation):
            for parameter in module.parameters():
                parameter.requires_grad = not freeze_permutation  # Enable gradients for LearnablePermutation
        else:
            for parameter in module.parameters():
                parameter.requires_grad = freeze_permutation  # Disable gradients for all other modules

def fgsm_attack(model, x, y, criterion, epsilon, device) -> torch.Tensor:
    """ Performs the Fast Gradient Sign Method (FGSM) attack, and returns the adversarial image

    Args:
        model (_type_): _description_
        x (_type_): _description_
        y (_type_): _description_
        criterion (_type_): _description_
        epsilon (_type_): _description_
        device (_type_): _description_

    Returns:
        torch.Tensor: _description_
    """
    x = x.to(device)
    y = y.to(device)
    model.to(device)
    
    x.requires_grad = True # Make sure x requires gradient
    logits = model(x)
    loss = criterion(logits, y)
    model.zero_grad() # Zero out previous gradients
    loss.backward()
    # Create adversarial examples using the sign of the gradients
    x_adv = x + epsilon * x.grad.sign()
    # Ensure the values are still within [0, 1]
    x_adv = torch.clamp(x_adv, 0, 1)
    
    return x_adv