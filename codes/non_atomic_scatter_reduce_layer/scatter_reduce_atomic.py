import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import warnings
import pytest
from typing import Annotated, Union, Callable, List
from utils import LearnablePermutation

def scatter_reduce_with_permutation(
    src: torch.Tensor,
    index: torch.Tensor,
    reduction: str = "sum"
    ) -> torch.tensor:
    """
    Custom scatter_reduce operation that permutes src and index based on a learnable permutation matrix.
    
    Args:
        src (torch.Tensor): The source tensor.
        index (torch.Tensor): The index tensor indicating where to scatter the values.
        perm_layer (LearnablePermutation): An instance of LearnablePermutation for generating the permutation matrix.
        reduction (str): The reduction operation ('sum', 'mean', etc.).
    
    Returns:
        torch.Tensor: The result of the scatter reduce operation after applying the permutation.
    """
    # Get the permutation matrix
    # perm_matrix = perm
    
    # Ensure src is of the same dtype as perm_matrix
    src = src.float()  # Convert src to float
    in_features = src.size(0)
    perm_matrix = LearnablePermutation(in_features).eval()()
    
    # Permute src
    # permuted_src = torch.matmul(perm_matrix, src.unsqueeze(1)).squeeze(1)
    permuted_src = torch.matmul(src, perm_matrix)
    permuted_index = torch.matmul(perm_matrix, index.unsqueeze(1).float()).squeeze(1).long()
    
    # Now proceed with the scatter operation
    if reduction == 'sum':
        result = torch.zeros_like(permuted_src)
        result = result.scatter_add(0, permuted_index, permuted_src)
    elif reduction == 'mean':
        result = torch.zeros_like(permuted_src)
        result = result.scatter_add(0, permuted_index, permuted_src)
        counts = torch.bincount(index, minlength=result.size(0))
        result = result / counts.clamp(min=1)
    
    return result

@pytest.mark.parametrize("src, index, reduction", [
    (torch.tensor([1.0, 2.0, 3.0, 4.0]), torch.tensor([0, 1, 0, 1], dtype=torch.long), 'sum'),
    (torch.tensor([5.0, 6.0, 7.0, 8.0]), torch.tensor([1, 0, 1, 0], dtype=torch.long), 'mean'),
    (torch.tensor([9.0, 10.0, 11.0, 12.0]), torch.tensor([2, 2, 0, 1], dtype=torch.long), 'sum'),
    (torch.tensor([13.0, 14.0, 15.0, 16.0]), torch.tensor([3, 3, 0, 1], dtype=torch.long), 'mean'),
])
def test_scatter_reduce_with_permutation(src, index, reduction):
    # Standard scatter_reduce output
    expected_result = torch.zeros_like(src)
    if reduction == 'sum':
        expected_result = expected_result.scatter_add(0, index, src)
    elif reduction == 'mean':
        expected_result = expected_result.scatter_add(0, index, src)
        counts = torch.bincount(index, minlength=expected_result.size(0))
        expected_result = expected_result / counts.clamp(min=1)
        
    # Custom scatter_reduce with permutation
    result = scatter_reduce_with_permutation(src, index, reduction)

    # Assert equality
    print(result, expected_result)
    assert torch.equal(result, expected_result), f"Test failed for src={src}, index={index}, reduction={reduction}"
