import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, Reddit, Amazon, TUDataset, Coauthor
# from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.data import DataLoader
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv
import os
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import argparse
from typing import Annotated, Union, Callable, List
from scipy.optimize import linear_sum_assignment
import warnings
from pytorch_lightning.strategies import DDPStrategy

class LearnablePermutation(torch.nn.Module):
    def __init__(self, in_features, temperature=1.0):
        super(LearnablePermutation, self).__init__()
        self.in_features = in_features
        self.temperature = temperature
        self.logits = torch.nn.Parameter(torch.randn(in_features, in_features))
        self.init_identity()

    def init_identity(self):
        with torch.no_grad():
            self.logits.data = torch.eye(self.in_features) * 0.1 + torch.randn(self.in_features, self.in_features) * 0.01
    
    def forward(self):
        if self.training:
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(self.logits)))
            perm_matrix = F.softmax((self.logits + gumbel_noise) / self.temperature, dim=-1)
        else:
            perm_matrix = torch.zeros_like(self.logits)
            row_indices, col_indices = linear_sum_assignment(-1 * self.logits.detach().cpu().numpy(), maximize=True)
            perm_matrix[row_indices, col_indices] = 1.0
            
        self._validate_permutation_matrix(perm_matrix)
        # print(perm_matrix) 
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
                parameter.requires_grad = freeze_permutation
                
# class GraphSAGE(nn.Module):
#     def __init__(
#         self,
#         in_channels: int = None,
#         hidden_channels: int = 16,
#         out_channels: int = None,
#     ) -> nn.Module:
        
#         super(GraphSAGE, self).__init__()
#         self.conv1 = SAGEConv(in_channels, hidden_channels)
#         self.conv2 = SAGEConv(hidden_channels, out_channels)

#     def forward(
#         self,
#         x: torch.Tensor,
#         edge_index: Annotated[torch.Tensor, "torch.Size([2, ..]), dtype=torch.int64"],
#         ) -> torch.Tensor:
        
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, p=0.5) 
#         x = self.conv2(x, edge_index)
#         return F.log_softmax(x, dim=1)

# class GraphSAGE(nn.Module):
#     def __init__(
#         self,
#         in_channels: int,
#         hidden_channels: int = 16,
#         out_channels: int = None,
#         num_nodes: int = None  # You need to know the total number of nodes for the permutation matrix
#     ):
#         super(GraphSAGE, self).__init__()
        
#         # Define two convolution layers
#         self.conv1 = SAGEConv(in_channels, hidden_channels)
#         self.conv2 = SAGEConv(hidden_channels, out_channels)

#         # Two separate learnable permutation layers for each convolution layer
#         self.permutation1 = LearnablePermutation(num_nodes)  # Permutation for conv1
#         self.permutation2 = LearnablePermutation(num_nodes)  # Permutation for conv2

#     def apply_permutation(self, edge_index, perm_matrix):
#         # Permute the source and destination nodes in edge_index with the learnable permutation matrix

#         # Convert the edge indices into one-hot encoding
#         num_nodes = perm_matrix.size(0)
        
#         # Create one-hot encoded vectors for source and target nodes
#         src_one_hot = F.one_hot(edge_index[0], num_classes=num_nodes).float()  # [num_edges, num_nodes]
#         tgt_one_hot = F.one_hot(edge_index[1], num_classes=num_nodes).float()  # [num_edges, num_nodes]
        
#         # Multiply the one-hot encoded vectors with the permutation matrix
#         permuted_src = torch.matmul(src_one_hot, perm_matrix)  # [num_edges, num_nodes]
#         permuted_tgt = torch.matmul(tgt_one_hot, perm_matrix)  # [num_edges, num_nodes]
        
#         # Convert the permuted one-hot vectors back to indices by taking the argmax
#         new_edge_index = torch.stack([
#             permuted_src.argmax(dim=1),  # Permuted source nodes
#             permuted_tgt.argmax(dim=1)   # Permuted target nodes
#         ], dim=0)
        
#         return new_edge_index

#     def forward(self, x, edge_index):
#         # Get the first learnable permutation matrix (for conv1)
#         perm_matrix1 = self.permutation1()

#         # Apply the first permutation matrix to the edge_index before the first convolution
#         edge_index_perm1 = self.apply_permutation(edge_index, perm_matrix1)

#         # First convolution layer
#         x = self.conv1(x, edge_index_perm1)
#         x = F.relu(x)
#         x = F.dropout(x, p=0.5, training=self.training)

#         # Get the second learnable permutation matrix (for conv2)
#         perm_matrix2 = self.permutation2()

#         # Apply the second permutation matrix to the edge_index before the second convolution
#         edge_index_perm2 = self.apply_permutation(edge_index, perm_matrix2)

#         # Second convolution layer
#         x = self.conv2(x, edge_index_perm2)

#         return F.log_softmax(x, dim=1)

class GraphSAGE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        out_channels: int = None,
        num_nodes: int = None,  # You need to know the total number of nodes for the permutation matrix
        num_layers: int = 2  # Number of convolutional layers
    ):
        super(GraphSAGE, self).__init__()

        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.permutations = nn.ModuleList()

        # Create the specified number of convolution layers and learnable permutation layers
        for i in range(num_layers):
            # Determine input and output channels for each layer
            if i == 0:
                self.convs.append(SAGEConv(in_channels, hidden_channels))
            elif i < num_layers - 1:
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            else:
                self.convs.append(SAGEConv(hidden_channels, out_channels))

            # Add a learnable permutation layer for this convolution layer
            self.permutations.append(LearnablePermutation(num_nodes))

    def apply_permutation(self, edge_index, perm_matrix):
        num_nodes = perm_matrix.size(0)

        # Create one-hot encoded vectors for source and target nodes
        src_one_hot = F.one_hot(edge_index[0], num_classes=num_nodes).float()
        tgt_one_hot = F.one_hot(edge_index[1], num_classes=num_nodes).float()

        # Multiply the one-hot encoded vectors with the permutation matrix
        permuted_src = torch.matmul(src_one_hot, perm_matrix)
        permuted_tgt = torch.matmul(tgt_one_hot, perm_matrix)

        # Convert the permuted one-hot vectors back to indices by taking the argmax
        new_edge_index = torch.stack([
            permuted_src.argmax(dim=1),  # Permuted source nodes
            permuted_tgt.argmax(dim=1)   # Permuted target nodes
        ], dim=0)

        return new_edge_index

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            # Get the current learnable permutation matrix
            perm_matrix = self.permutations[i]()
            
            # Apply the permutation to the edge_index
            edge_index_perm = self.apply_permutation(edge_index, perm_matrix)

            # Perform the convolution operation
            x = self.convs[i](x, edge_index_perm)
            x = F.relu(x)

            # Apply dropout for regularization
            x = F.dropout(x, p=0.5, training=self.training)

        return F.log_softmax(x, dim=1)

class GCN(nn.Module):
    def __init__(
        self,
        in_channels: int = None,
        hidden_channels: int = 32,
        out_channels: int = None,
        num_layers: int = 2,  # Number of convolutional layers
        num_nodes: int = None  # Total number of nodes for permutation
    ) -> nn.Module:
        super(GCN, self).__init__()

        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.permutations = nn.ModuleList()

        # Create the specified number of convolution layers and learnable permutation layers
        for i in range(num_layers):
            if i == 0:
                self.convs.append(GCNConv(in_channels, hidden_channels))
            elif i < num_layers - 1:
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
            else:
                self.convs.append(GCNConv(hidden_channels, out_channels))

            # Add a learnable permutation layer for this convolution layer
            self.permutations.append(LearnablePermutation(num_nodes))

    def apply_permutation(self, edge_index, perm_matrix):
        num_nodes = perm_matrix.size(0)

        # Create one-hot encoded vectors for source and target nodes
        src_one_hot = F.one_hot(edge_index[0], num_classes=num_nodes).float()
        tgt_one_hot = F.one_hot(edge_index[1], num_classes=num_nodes).float()

        # Multiply the one-hot encoded vectors with the permutation matrix
        permuted_src = torch.matmul(src_one_hot, perm_matrix)
        permuted_tgt = torch.matmul(tgt_one_hot, perm_matrix)

        # Convert the permuted one-hot vectors back to indices by taking the argmax
        new_edge_index = torch.stack([
            permuted_src.argmax(dim=1),  # Permuted source nodes
            permuted_tgt.argmax(dim=1)   # Permuted target nodes
        ], dim=0)

        return new_edge_index

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        for i in range(self.num_layers):
            # Get the current learnable permutation matrix
            perm_matrix = self.permutations[i]()

            # Apply the permutation to the edge_index
            edge_index_perm = self.apply_permutation(edge_index, perm_matrix)

            # Perform the convolution operation
            x = self.convs[i](x, edge_index_perm)
            x = F.relu(x)

            # Apply dropout for regularization
            x = F.dropout(x, p=0.5, training=self.training)

        return F.log_softmax(x, dim=1)
    
# class GCN(nn.Module):
#     def __init__(
#         self,
#         in_channels: int = None,
#         hidden_channels: int = 16,
#         out_channels: int = None,
#     ) -> nn.Module:
        
#         super(GCN, self).__init__()
#         self.conv1 = GCNConv(in_channels, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, out_channels)

#     def forward(
#         self,
#         x: torch.Tensor,
#         edge_index: Annotated[torch.Tensor, "torch.Size([2, ..]), dtype=torch.int64"],
#     ) -> torch.Tensor:
        
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, p=0.5)
#         x = self.conv2(x, edge_index)
#         return F.log_softmax(x, dim=1)

# class GAT(nn.Module):
#     def __init__(
#         self,
#         in_channels: int = None,
#         hidden_channels: int = 8,
#         out_channels: int = None,
#         heads: int = 8,
#     ) -> nn.Module:
        
#         super(GAT, self).__init__()
#         self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
#         self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False)

#     def forward(
#         self,
#         x: torch.Tensor,
#         edge_index: Annotated[torch.Tensor, "torch.Size([2, ..]), dtype=torch.int64"],
#     ) -> torch.Tensor:
        
#         x = self.conv1(x, edge_index)
#         x = F.elu(x)
#         x = F.dropout(x, p=0.5)
#         x = self.conv2(x, edge_index)
#         return F.log_softmax(x, dim=1)
    
class GAT(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 8,
        out_channels: int = None,
        heads: int = 8,
        num_layers: int = 2,  # Number of GAT layers
        num_nodes: int = None  # Total number of nodes for the permutation
    ):
        super(GAT, self).__init__()

        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.permutations = nn.ModuleList()

        # Create GAT layers and learnable permutation layers
        for i in range(num_layers):
            if i == 0:
                # First GAT layer
                self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, concat=True))
            elif i < num_layers - 1:
                # Intermediate GAT layers
                self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True))
            else:
                # Last GAT layer
                self.convs.append(GATConv(hidden_channels * heads, out_channels, heads=1, concat=False))

            # Add a learnable permutation layer for this convolution layer
            self.permutations.append(LearnablePermutation(num_nodes))

    def apply_permutation(self, edge_index, perm_matrix):
        num_nodes = perm_matrix.size(0)

        # Create one-hot encoded vectors for source and target nodes
        src_one_hot = F.one_hot(edge_index[0], num_classes=num_nodes).float()
        tgt_one_hot = F.one_hot(edge_index[1], num_classes=num_nodes).float()

        # Multiply the one-hot encoded vectors with the permutation matrix
        permuted_src = torch.matmul(src_one_hot, perm_matrix)
        permuted_tgt = torch.matmul(tgt_one_hot, perm_matrix)

        # Convert the permuted one-hot vectors back to indices by taking the argmax
        new_edge_index = torch.stack([
            permuted_src.argmax(dim=1),  # Permuted source nodes
            permuted_tgt.argmax(dim=1)   # Permuted target nodes
        ], dim=0)

        return new_edge_index

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        for i in range(self.num_layers):
            # Get the current learnable permutation matrix
            perm_matrix = self.permutations[i]()

            # Apply the permutation to the edge_index
            edge_index_perm = self.apply_permutation(edge_index, perm_matrix)

            # Perform the GAT convolution operation
            x = self.convs[i](x, edge_index_perm)
            x = F.elu(x)

            # Apply dropout for regularization
            x = F.dropout(x, p=0.5, training=self.training)

        return F.log_softmax(x, dim=1)

class GIN(nn.Module):
    def __init__(
        self,
        in_channels: int = None,
        hidden_channels: int = 16,
        out_channels: int = None,
    ) -> nn.Module:
        
        super(GIN, self).__init__()
        nn1 = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.conv1 = GINConv(nn1)

        nn2 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )
        self.conv2 = GINConv(nn2)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Annotated[torch.Tensor, "torch.Size([2, ..]), dtype=torch.int64"],
    ) -> torch.Tensor:
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
    
# class PinSAGE(nn.Module):
#     def __init__(
#         self,
#         in_channels: int = None,
#         hidden_channels: int = 64,
#         out_channels: int = None,
#     ) -> nn.Module:
        
#         super(PinSAGE, self).__init__()
#         self.conv1 = SAGEConv(in_channels, hidden_channels)
#         self.conv2 = SAGEConv(hidden_channels, out_channels)

#     def forward(
#         self,
#         x: torch.Tensor,
#         edge_index: Annotated[torch.Tensor, "torch.Size([2, ..]), dtype=torch.int64"],
#     ) -> torch.Tensor:
        
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, p=0.5)
#         x = self.conv2(x, edge_index)
#         return F.log_softmax(x, dim=1)

class GIN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 16,
        out_channels: int = None,
        num_layers: int = 2,  # Number of GIN layers
        num_nodes: int = None  # Total number of nodes for the permutation
    ):
        super(GIN, self).__init__()

        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.permutations = nn.ModuleList()

        # Initialize GIN layers and permutation layers
        for i in range(num_layers):
            if i == 0:
                nn_block = nn.Sequential(
                    nn.Linear(in_channels, hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, hidden_channels)
                )
                self.convs.append(GINConv(nn_block))
            elif i < num_layers - 1:
                nn_block = nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, hidden_channels)
                )
                self.convs.append(GINConv(nn_block))
            else:
                nn_block = nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, out_channels)
                )
                self.convs.append(GINConv(nn_block))

            # Add a learnable permutation matrix for this layer
            self.permutations.append(LearnablePermutation(num_nodes))

    def apply_permutation(self, edge_index, perm_matrix):
        num_nodes = perm_matrix.size(0)

        # Create one-hot encoded vectors for source and target nodes
        src_one_hot = F.one_hot(edge_index[0], num_classes=num_nodes).float()
        tgt_one_hot = F.one_hot(edge_index[1], num_classes=num_nodes).float()

        # Multiply the one-hot encoded vectors with the permutation matrix
        permuted_src = torch.matmul(src_one_hot, perm_matrix)
        permuted_tgt = torch.matmul(tgt_one_hot, perm_matrix)

        # Convert the permuted one-hot vectors back to indices by taking the argmax
        new_edge_index = torch.stack([
            permuted_src.argmax(dim=1),  # Permuted source nodes
            permuted_tgt.argmax(dim=1)   # Permuted target nodes
        ], dim=0)

        return new_edge_index

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i in range(self.num_layers):
            # Get the current learnable permutation matrix
            perm_matrix = self.permutations[i]()

            # Apply the permutation to the edge_index
            edge_index_perm = self.apply_permutation(edge_index, perm_matrix)

            # Perform the GIN convolution
            x = self.convs[i](x, edge_index_perm)
            x = F.relu(x)

            # Apply dropout for regularization
            x = F.dropout(x, p=0.5, training=self.training)

        return F.log_softmax(x, dim=1)
    
class PinSAGE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = None,
        num_layers: int = 2,  # Number of SAGE layers
        num_nodes: int = None  # Total number of nodes for the permutation
    ) -> nn.Module:
        super(PinSAGE, self).__init__()

        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.permutations = nn.ModuleList()

        # Initialize SAGE layers and permutation layers
        for i in range(num_layers):
            if i == 0:
                self.convs.append(SAGEConv(in_channels, hidden_channels))
            elif i < num_layers - 1:
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            else:
                self.convs.append(SAGEConv(hidden_channels, out_channels))

            # Add a learnable permutation matrix for this layer
            self.permutations.append(LearnablePermutation(num_nodes))

    def apply_permutation(self, edge_index, perm_matrix):
        num_nodes = perm_matrix.size(0)

        # Create one-hot encoded vectors for source and target nodes
        src_one_hot = F.one_hot(edge_index[0], num_classes=num_nodes).float()
        tgt_one_hot = F.one_hot(edge_index[1], num_classes=num_nodes).float()

        # Multiply the one-hot encoded vectors with the permutation matrix
        permuted_src = torch.matmul(src_one_hot, perm_matrix)
        permuted_tgt = torch.matmul(tgt_one_hot, perm_matrix)

        # Convert the permuted one-hot vectors back to indices by taking the argmax
        new_edge_index = torch.stack([
            permuted_src.argmax(dim=1),  # Permuted source nodes
            permuted_tgt.argmax(dim=1)   # Permuted target nodes
        ], dim=0)

        return new_edge_index

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i in range(self.num_layers):
            # Get the current learnable permutation matrix
            perm_matrix = self.permutations[i]()

            # Apply the permutation to the edge_index
            edge_index_perm = self.apply_permutation(edge_index, perm_matrix)

            # Perform the SAGE convolution
            x = self.convs[i](x, edge_index_perm)
            x = F.relu(x)

            # Apply dropout for regularization
            x = F.dropout(x, p=0.5, training=self.training)

        return F.log_softmax(x, dim=1)
    
class LightningClassifierGNN(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        criterion: torch.nn = nn.CrossEntropyLoss(),
        optimiser: torch.optim = torch.optim.SGD,
        perm_optimiser: torch.optim.Optimizer = None,
        deterministic_train: bool = True,
        deterministic_test: bool = True,
        deterministic_attack: bool = True
    ) -> nn.Module:
        super(LightningClassifierGNN, self).__init__()

        self.model = model
        self.criterion = criterion
        self.optimiser = optimiser
        
        try: 
            self.perm_optimiser =  torch.optim.SGD(
                [param for name, param in self.model.named_parameters() if "permutation" in name], 
                lr=1e-3
            )
        except:
           self.perm_optimiser = None 
        
        self.deterministic_train = deterministic_train
        self.deterministic_test = deterministic_test
        self.deterministic_attack = deterministic_attack

    def loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> float:
        return self.criterion(logits, labels)

    def training_step(
        self,
        train_batch: Union[Batch, Data],
        batch_idx: int
    ) -> float:
        
        # print(f"train_batch: {train_batch}, batch_idx: {batch_idx}")
        
        torch.use_deterministic_algorithms(self.deterministic_train)
        
        data = train_batch
        
        logits = self.model(data.x, data.edge_index)
        loss = self.loss(logits[data.train_mask], data.y[data.train_mask])
        
        self.log("train_loss", loss)
        
        return loss

    def validation_step(
        self,
        val_batch: Union[Batch, Data],
        batch_idx: int
    ) -> None:
        
        torch.use_deterministic_algorithms(self.deterministic_test)

        data = val_batch

        logits = self.model(data.x, data.edge_index)
        loss = self.loss(logits[data.val_mask], data.y[data.val_mask])

        self.log("val_loss", loss)

    def configure_optimizers(
        self
    ) -> torch.optim:
        return self.optimiser(self.parameters(), lr=1e-3)

    def adversarial_attack(
        self,
        attack_fn: Callable,
        epsilon_list: List[float],
        device: torch.device,
        log_path: Path,
        iter: int
    ) -> None:
        
        data_loader = self.trainer.datamodule.val_dataloader()
        torch.use_deterministic_algorithms(self.deterministic_attack)

        # Set model in evaluation mode
        model = self.model.to(device)
        model.eval()

        results: List = list([])

        for epsilon in tqdm(epsilon_list):
            # baseline_correct = 0
            attack_correct = 0
            total_samples = 0
            for data in data_loader:
                data = data.to(device)
                
                with torch.no_grad():
                    baseline_logits = self.model(data.x, data.edge_index)

                    # Baseline accuracy before attack
                    baseline_preds = torch.argmax(baseline_logits, dim=1)
                    # baseline_correct += (baseline_preds[data.val_mask] == data.y[data.val_mask]).sum().item()

                # Generate adversarial examples using the passed attack function
                attack_data = attack_fn(model, data, self.criterion, epsilon, device)
                attack_logits = model(attack_data.x, attack_data.edge_index)

                # Compute accuracy on adversarial examples
                attack_preds = torch.argmax(attack_logits, dim=1)
                # attack_correct += (attack_preds[data.val_mask] == data.y[data.val_mask]).sum().item()
                attack_correct += (attack_preds[data.val_mask] == baseline_preds[data.val_mask]).sum().item() 

                total_samples += data.val_mask.sum().item()  # Total valid samples in this batch
            
            # baseline_accuracy = baseline_correct / len(data_loader.dataset)
            attack_accuracy = attack_correct / total_samples

            # # Log results using the logger
            # logger.log_metrics({
            #     f'baseline_accuracy_epsilon_{epsilon}': baseline_accuracy,
            #     f'attack_accuracy_epsilon_{epsilon}': attack_accuracy
            # })
            
            results.append({
                "epsilon": epsilon,
                # "baseline_accuracy": baseline_accuracy,
                "accuracy": attack_accuracy,
                "total_correct": f"{attack_correct}/{total_samples}"
            })

        # Saving results in CSV using logger
        results_df = pd.DataFrame(results)
        # csv_save_path = os.path.join(logger.log_dir, 'adversarial_attack_results.csv')
        results_df.to_csv(f"{log_path}/{attack_fn.__name__}_{iter}.csv", index=False)
        # results_df.to_csv(csv_save_path, index=False)

    def maximize_loss_fixed_input(
        self,
        model: nn.Module,
        data: Batch,  # Expecting a Batch object containing graph data
        optimizer: torch.optim,
        criterion: torch.nn,
        device: torch.device,
        iterations: int = 10
    ) -> int:
        model.to(device)
        self.model.train()
        torch.use_deterministic_algorithms(self.deterministic_attack)
        freeze_permutations(self.model, False)

        # Move graph data to device
        data = data.to(device)

        # Extract input features and labels
        inputs = data.x
        labels = data.y

        # Perform optimization
        for iteration in range(iterations):
            self.perm_optimiser.zero_grad()
            
            # Forward pass through the model
            logits = model(inputs, data.edge_index)  # Pass the graph data
            
            loss = self.criterion(logits[data.val_mask], labels[data.val_mask])  # Calculate loss only for valid nodes
            loss = -loss  # Maximize loss
            loss.backward(retain_graph=True)
            
            self.perm_optimiser.step()

        # # Check if any classification has flipped
        # preds = torch.argmax(logits, dim=1)
        # flipped_count = (preds[data.val_mask] != labels[data.val_mask]).sum().item()  # Count how many labels are flipped

        # if flipped_count > 0:
        #     print(f"Classification Flip! Flipped Count: {flipped_count}")
        #     return flipped_count

        return logits
    
    def maximize_loss_fixed_adversarial_input(
        self,
        epsilon_list: List[float],
        device: torch.device,
        attack_fn: Callable,
        log_path: Path
    ) -> None:
        
        torch.use_deterministic_algorithms(self.deterministic_attack)
        model = self.model.to(device)
        data_loader = self.trainer.datamodule.val_dataloader()
        results: List = []

        for epsilon in tqdm(epsilon_list):
            baseline_correct: int = 0
            attack_correct: int = 0
            perm_correct: int = 0  # Initialize permutation correct count
            total_samples: int = 0  # Initialize total sample count
            
            for data in tqdm(data_loader):
                data = data.to(device)  # Move graph data to device

                # Generate adversarial examples
                attack_data = attack_fn(model, data, self.criterion, epsilon, device)

                with torch.no_grad():
                    # Compute baseline predictions on adversarial inputs
                    baseline_logits = model(data.x, attack_data.edge_index)
                    baseline_preds = torch.argmax(baseline_logits, dim=1)

                    # Compute attack predictions on adversarial inputs
                    attack_logits = model(attack_data.x, attack_data.edge_index)
                    attack_preds = torch.argmax(attack_logits, dim=1)

                # Count attack accuracy
                attack_correct += (attack_preds[data.val_mask] == baseline_preds[data.val_mask]).sum().item()

                # Optimize model using adversarial inputs
                perm_logits = self.maximize_loss_fixed_input(
                    model, attack_data, self.perm_optimiser, self.criterion, device, iterations=10
                )
                
                # We can also evaluate the permuted logits here if needed for further analysis.
                perm_preds = torch.argmax(perm_logits, dim=1)
                perm_correct += (perm_preds[data.val_mask] == baseline_preds[data.val_mask]).sum().item()  # Count correct predictions

                total_samples += data.val_mask.sum().item()

            # Store results for the current epsilon
            perm_correct = min(attack_correct, perm_correct)
            
            results.append({
                "epsilon": epsilon,
                "baseline_accuracy": attack_correct / total_samples if total_samples > 0 else 0.0,
                "attack_accuracy": perm_correct / total_samples if total_samples > 0 else 0.0,
                "attack_correct": f"{attack_correct}/{total_samples}",  # Store the correct count
                "perm_correct": f"{perm_correct}/{total_samples}",  # Store the correct count
            })
            
        # Save results to CSV
        df = pd.DataFrame(results)
        df.to_csv(f"{log_path}/{attack_fn.__name__}.csv", index=False)

    # def maximize_loss_fixed_adversarial_input(
    #     self,
    #     epsilon_list: List[float],
    #     device: torch.device,
    #     attack_fn: Callable,
    #     log_path: Path
    # ) -> None:
        
    #     torch.use_deterministic_algorithms(self.deterministic_attack)
    #     model = self.model.to(device)
    #     data_loader = self.trainer.datamodule.val_dataloader()
    #     results: List = list([])

    #     for epsilon in tqdm(epsilon_list):
    #         baseline_correct: int = 0
    #         permutation_correct: int = 0
    #         total_samples: int = 0  # Initialize total sample count
            
    #         for data in tqdm(data_loader):
    #             data = data.to(device)  # Move graph data to device
                
    #             # Compute baseline predictions
    #             with torch.no_grad():
    #                 baseline_logits = model(data.x, data.edge_index)
    #                 baseline_preds = torch.argmax(baseline_logits, dim=1)
                    
    #                 # Calculate baseline accuracy
    #                 baseline_correct += (baseline_preds[data.val_mask] == data.y[data.val_mask]).sum().item()
    #                 total_samples += data.val_mask.sum().item()  # Count total valid nodes
                
    #             # Generate adversarial examples
    #             attack_data = attack_fn(model, data, self.criterion, epsilon, device)
                
    #             with torch.no_grad():
    #                 attack_logits = model(attack_data.x, attack_data.edge_index)
    #                 # Evaluate the attack
    #                 attack_preds = torch.argmax(attack_logits, dim=1)
                    
    #             perm_logits = self.maximize_loss_fixed_input(
    #                 self.model, attack_data, self.perm_optimiser, self.criterion, device, iterations=10
    #             )
                
    #             perm_preds = torch.argmax(perm_logits, dim=1)
    #             permutation_correct += (perm_preds[data.val_mask] == data.y[data.val_mask]).sum().item()


    #             # Store results
    #             results.append({
    #                 "epsilon": epsilon,
    #                 "baseline_accuracy": baseline_correct / total_samples * 100 if total_samples > 0 else 0.0,
    #                 "permutation_accuracy": permutation_correct / total_samples * 100 if total_samples > 0 else 0.0,
    #                 "total_correct": f"{baseline_correct}/{total_samples}"  # Store the correct count
    #             })
                
    #     # Save results to CSV
    #     df = pd.DataFrame(results)
    #     df.to_csv(f"{log_path}/{attack_fn.__name__}.csv", index=False)


def fgsm_attack(
    model: nn.Module,
    data: Union[Batch, Data],
    criterion: torch.nn,
    epsilon: float,
    device: torch.device
    ) -> Union[Batch, Data]:
    
    attack_data = data.clone()
    attack_data.x.requires_grad = True

    logits = model(attack_data.x.to(device), attack_data.edge_index.to(device))
    loss = criterion(logits[data.train_mask], data.y[data.train_mask])
    
    model.zero_grad()
    loss.backward()

    attack_data.x = attack_data.x + epsilon * attack_data.x.grad.sign()
    return attack_data

def pgd_attack(
    model: nn.Module,
    data: Union[Batch, Data],
    criterion: torch.nn,
    epsilon: float,
    device: torch.device,
    alpha: float = 1.0,
    num_steps: int = 10,
) -> Union[Batch, Data]:
    
    # Clone the original data and enable gradient tracking
    attack_data = data.clone()
    attack_data.x.requires_grad = True

    # Save the original data features to project perturbations later
    original_x = attack_data.x.clone().detach()

    for _ in range(num_steps):
        # Forward pass: Compute logits and loss
        logits = model(attack_data.x.to(device), attack_data.edge_index.to(device))
        loss = criterion(logits[data.train_mask], data.y[data.train_mask])
        
        # Zero out previous gradients
        model.zero_grad()
        
        # Backward pass: Compute gradients with respect to input features
        loss.backward()

        # Apply gradient ascent (move in the direction of the gradient)
        attack_data.x = attack_data.x + alpha * attack_data.x.grad.sign()

        # Ensure the perturbation stays within the epsilon ball around the original input
        perturbation = torch.clamp(attack_data.x - original_x, min=-epsilon, max=epsilon)
        attack_data.x = torch.clamp(original_x + perturbation, min=0, max=1)

        # Detach the gradients after each step to prevent accumulation
        attack_data.x = attack_data.x.detach()
        attack_data.x.requires_grad = True

    return attack_data

def random_attack(
    model: nn.Module,
    data: Union[Batch, Data],
    criterion: torch.nn,
    epsilon: float,
    device: torch.device,
) -> Union[Batch, Data]:

    # Clone the original data
    attack_data = data.clone()

    # Generate random noise within the [-epsilon, epsilon] range
    random_noise = (2 * torch.rand_like(attack_data.x) - 1) * epsilon

    # Add the random noise to the original input
    attack_data.x = torch.clamp(attack_data.x + random_noise, min=0, max=1)

    return attack_data

def targeted_class_confidence_attack(
    model: nn.Module,
    data: Union[Batch, Data],
    criterion: torch.nn,
    epsilon: float,
    device: torch.device,
    alpha: float = 1.0,
    num_steps: int = 10,
) -> Union[Batch, Data]:

    # Clone the original data and enable gradient tracking
    attack_data = data.clone()
    attack_data.x.requires_grad = True

    # Save the original data features for projection
    original_x = attack_data.x.clone().detach()

    for _ in range(num_steps):
        # Forward pass: Compute logits
        logits = model(attack_data.x.to(device), attack_data.edge_index.to(device))

        # For each data point in the batch, find the most confident (max logit) and second most confident class
        top_logits = torch.topk(logits[data.train_mask], 2, dim=1)

        # Get indices of most and second most confident classes
        most_confident_class = top_logits.indices[:, 0]
        second_confident_class = top_logits.indices[:, 1]

        # Compute the difference between the logits of the most confident and second most confident classes
        targeted_loss = logits[data.train_mask, most_confident_class] - logits[data.train_mask, second_confident_class]

        # Minimize this difference to reduce the confidence of the most likely class and increase the second most likely
        loss = torch.mean(targeted_loss)
        
        # Zero out previous gradients
        model.zero_grad()

        # Backward pass: Compute gradients
        loss.backward()

        # Apply gradient ascent to maximize the second class's confidence and minimize the first class's confidence
        attack_data.x = attack_data.x - alpha * attack_data.x.grad.sign()

        # Project perturbation to keep it within the epsilon ball around the original input
        perturbation = torch.clamp(attack_data.x - original_x, min=-epsilon, max=epsilon)
        attack_data.x = torch.clamp(original_x + perturbation, min=0, max=1)

        # Detach gradients after each step to avoid accumulation
        attack_data.x = attack_data.x.detach()
        attack_data.x.requires_grad = True

    return attack_data


class LightningDataModuleGNN(pl.LightningDataModule):
    def __init__(self, dataset, batch_size):
        super(LightningDataModuleGNN, self).__init__()
        self.dataset = dataset
        self.batch_size = batch_size

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)

def CORA_data_module(batch_size):
    dataset = Planetoid(root='data/Planetoid', name='Cora', transform=T.NormalizeFeatures())
    data_module = LightningDataModuleGNN(dataset, batch_size)
    return data_module

def CiteSeer_data_module(batch_size):
    dataset = Planetoid(root='data/Planetoid', name='CiteSeer', transform=T.NormalizeFeatures())
    data_module = LightningDataModuleGNN(dataset, batch_size)
    return data_module

def PubMed_data_module(batch_size):
    dataset = Planetoid(root='data/Planetoid', name='PubMed', transform=T.NormalizeFeatures())
    data_module = LightningDataModuleGNN(dataset, batch_size)
    return data_module

def Reddit_data_module(batch_size):
    dataset = Reddit(root='data/Reddit')
    data_module = LightningDataModuleGNN(dataset, batch_size)
    return data_module

def AmazonComputers_data_module(batch_size):
    dataset = Amazon(root='data/Amazon', name='Computers', transform=T.NormalizeFeatures())
    data_module = LightningDataModuleGNN(dataset, batch_size)
    return data_module

def CoauthorCS_data_module(batch_size):
    dataset = Coauthor(root='data/Coauthor', name='CS', transform=T.NormalizeFeatures())
    data_module = LightningDataModuleGNN(dataset, batch_size)
    return data_module

def TUDataset_data_module(batch_size):
    dataset = TUDataset(root='data/TUDataset', name='ENZYMES', transform=T.NormalizeFeatures())
    data_module = LightningDataModuleGNN(dataset, batch_size)
    return data_module

def main(args):
    
    dataset_dispatcher: dict = {
        "CORA": CORA_data_module,
        "CiteSeer": CiteSeer_data_module,
        "PubMed": PubMed_data_module,
        "Reddit": Reddit_data_module,
        "AmazonComputers": AmazonComputers_data_module,
        "CoauthorCS": CoauthorCS_data_module,
        "TUDataset": TUDataset_data_module,
        # "ASTRO": get_astro_data_module,
    }
    
    attack_dispatcher: dict = {
        "fgsm_attack": fgsm_attack,
        "pgd_attack": pgd_attack,
        "random_attack": random_attack,
        "targeted_class_confidence_attack": targeted_class_confidence_attack
    }
    
    model_dispatcher: dict = {
        "GraphSAGE": GraphSAGE,
        "GCN": GCN,
        "GAT": GAT,
        "PinSAGE": PinSAGE,
        "GIN": GIN,
        # "MNIST": MNISTClassifier,
        # "MNISTLearnablePermutation": MNISTClassifierAtomicLinearLearnablePermutation,
        # "AtomicMNIST": AtomicMNISTClassifier,
        # "ASTRO": AstroClassifier,
        # "AtomicASTRO": AtomicAstroClassifier,
    }

    data_module = dataset_dispatcher[args.dataset](args.batch_size)
    
    model = LightningClassifierGNN(
        model = model_dispatcher[args.model](
                    in_channels=data_module.dataset.num_node_features,
                    out_channels=data_module.dataset.num_classes,
                    num_nodes=data_module.dataset[0].num_nodes,
                    num_layers=args.num_layers
                ),
        deterministic_train = True,
        deterministic_test = True,
        deterministic_attack = True
    )
    
    is_learable_permutation: bool = any(
        "permutation" in name for name, _ in model.named_parameters()
    )
    

    # model = GraphSAGE(in_channels=data_module.dataset.num_node_features,
    #                   hidden_channels=16,
    #                   out_channels=data_module.dataset.num_classes)

    # lightning_model = LightningClassifierGNN(model)

    # Ensure the logging directory exists
    # if args.log_dir is None:
    #     log_dir = "csv_logs"
    # else:
    #     log_dir = args.log_dir
    
    log_dir: Path 
    if args.log_dir: 
        log_dir = Path(args.log_dir)
    else:
        log_dir = Path(f"{torch.cuda.get_device_name()}/CSV_Logs")
        
    # log_path = log_dir / f"{args.experiment_name}" / f"version_{trainer.logger.version}"
    
    csv_logger = pl.loggers.CSVLogger(
        save_dir=log_dir, 
        name=args.experiment_name
    )

    # # CSV logger for both training and adversarial attack logs
    # csv_logger = pl.loggers.CSVLogger(log_dir, name=args.experiment_name)

    # Trainer setup
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=csv_logger,
        devices=args.devices,
        strategy=DDPStrategy(find_unused_parameters=True)
    )

    log_path = log_dir / f"{args.experiment_name}" / f"version_{trainer.logger.version}"
    
    # Train the model
    trainer.fit(model, data_module)

    # # Run adversarial attacks after training
    
    if args.num_inference_runs:
        for iter in range(args.num_inference_runs):
            # model.adversarial_attack(
            #     attack_fn=fgsm_attack,
            #     epsilon_list=[0, 0.00001, 0.0001, 0.001, 0.01, 0.1],
            #     log_path=log_path,
            #     device=torch.device("cuda"),
            #     iter=iter
            # )
            # model.adversarial_attack(
            #     attack_fn=pgd_attack,
            #     epsilon_list=[0, 0.00001, 0.0001, 0.001, 0.01, 0.1],
            #     log_path=log_path,
            #     device=torch.device("cuda"),
            #     iter=iter
            # )
            # model.adversarial_attack(
            #     attack_fn=random_attack,
            #     epsilon_list=[0, 0.00001, 0.0001, 0.001, 0.01, 0.1],
            #     log_path=log_path,
            #     device=torch.device("cuda"),
            #     iter=iter
            # )
            model.adversarial_attack(
                attack_fn=targeted_class_confidence_attack,
                epsilon_list=[0, 0.00001, 0.0001, 0.001, 0.01, 0.1],
                log_path=log_path,
                device=torch.device("cuda"),
                iter=iter
            )
    
    if is_learable_permutation:
        # model.maximize_loss_fixed_adversarial_input(
        #     attack_fn=fgsm_attack,
        #     epsilon_list=[0, 0.00001, 0.0001, 0.001, 0.01, 0.1],
        #     log_path=log_path,
        #     device=torch.device("cuda")
        # )
        # model.maximize_loss_fixed_adversarial_input(
        #     attack_fn=pgd_attack,
        #     epsilon_list=[0, 0.00001, 0.0001, 0.001, 0.01, 0.1],
        #     log_path=log_path,
        #     device=torch.device("cuda")
        # )
        # model.maximize_loss_fixed_adversarial_input(
        #     attack_fn=random_attack,
        #     epsilon_list=[0, 0.00001, 0.0001, 0.001, 0.01, 0.1],
        #     log_path=log_path,
        #     device=torch.device("cuda")
        # )
        model.maximize_loss_fixed_adversarial_input(
            attack_fn=targeted_class_confidence_attack,
            epsilon_list=[0, 0.00001, 0.0001, 0.001, 0.01, 0.1],
            log_path=log_path,
            device=torch.device("cuda")
        )


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Train and evaluate a GraphSAGE model with PyTorch Lightning"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset to classify",
        required=True
    )
    
    parser.add_argument(
        "--model",
        type=str,
        help="Model to apply to dataset",
        required=True
    )
    
    parser.add_argument(
        "--attack",
        type=str,
        help="Model to apply to dataset",
        default="targeted_class_confidence_attack"
    )
    
    parser.add_argument(
        "--num_layers",
        type=int,
        help="Number of layers in GNN",
        default=10
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for training and testing"
    )
    
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=25,
        help="Number of epochs for training"
    )
    
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="gnn",
        help="Name of the experiment for logging",
    )
    
    parser.add_argument(
        "--log_dir",
        type=str,
        # default="csv_logs",
        required=False,
        help="Directory to save logs"
    )
    
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of GPUs to use (0 for CPU)"
    )
    
    parser.add_argument(
        "--num_inference_runs",
        type=int,
        default=10,
        required=False,
        help="Number of GPUs to use (0 for CPU)"
    )
    
    parser.add_argument(
        "--deterministic_train",
        action='store_true',
        help="Set to True for deterministic training. Default is False."
    )

    parser.add_argument(
        "--deterministic_test",
        action='store_true',
        help="Set to True for deterministic testing. Default is False."
    )

    parser.add_argument(
        "--deterministic_attack",
        action='store_true',
        help="Set to True for deterministic attack. Default is False."
    )

    args = parser.parse_args()
    
    main(args)
