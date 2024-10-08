import torch
import pytest
from torch_geometric.nn import SAGEConv as PyG_SAGEConv

class LearnablePermutation(torch.nn.Module):
    def __init__(self, in_features):
        super(LearnablePermutation, self).__init__()
        self.perm_matrix = torch.nn.Parameter(torch.eye(in_features))

    def forward(self):
        return self.perm_matrix

def scatter_reduce_with_permutation(
    src: torch.Tensor,
    index: torch.Tensor,
    reduction: str = "sum"
) -> torch.Tensor:
    src = src.float()  # Ensure src is float
    num_nodes = src.size(0)  # Number of nodes
    in_features = src.size(1)  # Number of input features

    # Get the learnable permutation matrix
    perm_layer = LearnablePermutation(in_features)
    perm_matrix = perm_layer()  # Shape: [in_features, in_features]

    # Permute src
    permuted_src = torch.matmul(src, perm_matrix)  # Shape: [num_nodes, in_features]

    # Scatter reduce based on the specified reduction operation
    if reduction == 'sum':
        result = torch.zeros(num_nodes, in_features, device=src.device)  # Result shape: [num_nodes, in_features]
        # We need to ensure that `index` has the same length as `permuted_src`
        result.index_add_(0, index, permuted_src[index])  # Scatter add based on index
    elif reduction == 'mean':
        result = torch.zeros(num_nodes, in_features, device=src.device)
        result.index_add_(0, index, permuted_src[index])  # Scatter add based on index
        counts = torch.bincount(index, minlength=num_nodes)
        result = result / counts.clamp(min=1).unsqueeze(1)  # Avoid division by zero

    return result

class CustomSAGEConv(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(CustomSAGEConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.Tensor(in_features, out_features))
        self.permutation_layer = LearnablePermutation(in_features)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Get the learnable permutation matrix
        perm_matrix = self.permutation_layer()  # Shape: [in_features, in_features]
        permuted_x = torch.matmul(x, perm_matrix)  # Permute features: [num_nodes, in_features]

        row, col = edge_index  # Edge index: row and column indices
        
        # Aggregation step using the scatter reduce with permutation
        aggregated = scatter_reduce_with_permutation(permuted_x, row, reduction='mean')

        # Output transformation
        out = torch.matmul(aggregated, self.weight)  # Shape: [num_nodes, out_features]
        return out

# Test function to compare CustomSAGEConv with PyG SAGEConv
@pytest.mark.parametrize("num_nodes, in_features, out_features, num_edges", [
    (100, 16, 32, 300),
])
def test_sageconv(num_nodes, in_features, out_features, num_edges):
    # Create random input feature matrix and edge index
    x = torch.randn(num_nodes, in_features)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))

    # Initialize both SAGEConv layers
    custom_sage_conv = CustomSAGEConv(in_features, out_features)
    pyg_sage_conv = PyG_SAGEConv(in_features, out_features)

    # Forward pass with the custom SAGEConv
    custom_output = custom_sage_conv(x, edge_index)

    # Forward pass with the PyG SAGEConv
    pyg_output = pyg_sage_conv(x, edge_index)

    # Check if outputs are close
    assert torch.allclose(custom_output, pyg_output, atol=1e-6), "Outputs do not match!"

if __name__ == "__main__":
    pytest.main()
