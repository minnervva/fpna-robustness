import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import warnings
import pytest

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
        print(perm_matrix) 
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