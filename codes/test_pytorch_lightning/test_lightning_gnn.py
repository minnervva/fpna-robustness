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


class GraphSAGE(nn.Module):
    def __init__(
        self,
        in_channels: int = None,
        hidden_channels: int = 16,
        out_channels: int = None,
    ) -> nn.Module:
        
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

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

class GCN(nn.Module):
    def __init__(
        self,
        in_channels: int = None,
        hidden_channels: int = 16,
        out_channels: int = None,
    ) -> nn.Module:
        
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

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

class GAT(nn.Module):
    def __init__(
        self,
        in_channels: int = None,
        hidden_channels: int = 8,
        out_channels: int = None,
        heads: int = 8,
    ) -> nn.Module:
        
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Annotated[torch.Tensor, "torch.Size([2, ..]), dtype=torch.int64"],
    ) -> torch.Tensor:
        
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.5)
        x = self.conv2(x, edge_index)
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
    
class PinSAGE(nn.Module):
    def __init__(
        self,
        in_channels: int = None,
        hidden_channels: int = 64,
        out_channels: int = None,
    ) -> nn.Module:
        
        super(PinSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

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
        log_path: Path
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
            for data in data_loader:
                data = data.to(device)
                
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

            # baseline_accuracy = baseline_correct / len(data_loader.dataset)
            attack_accuracy = attack_correct / len(data_loader)

            # # Log results using the logger
            # logger.log_metrics({
            #     f'baseline_accuracy_epsilon_{epsilon}': baseline_accuracy,
            #     f'attack_accuracy_epsilon_{epsilon}': attack_accuracy
            # })
            
            results.append({
                "epsilon": epsilon,
                # "baseline_accuracy": baseline_accuracy,
                "accuracy": attack_accuracy
            })

        # Saving results in CSV using logger
        results_df = pd.DataFrame(results)
        # csv_save_path = os.path.join(logger.log_dir, 'adversarial_attack_results.csv')
        results_df.to_csv(f"{log_path}/{attack_fn.__name__}.csv", index=False)
        # results_df.to_csv(csv_save_path, index=False)


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

# class LightningDataModuleGNN(pl.LightningDataModule):
#     def __init__(self, dataset, batch_size):
#         super(LightningDataModuleGNN, self).__init__()
#         self.dataset = dataset
#         self.batch_size = batch_size

#     def setup(self, stage=None):
#         transform = T.NormalizeFeatures()
#         self.dataset = Planetoid(root='data/Planetoid', name='Cora', transform=transform)

#     def train_dataloader(self):
#         return GeoDataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

#     def val_dataloader(self):
#         return GeoDataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)

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
    
    dataset_dispatcher = {
        "CORA": CORA_data_module,
        "CiteSeer": CiteSeer_data_module,
        "PubMed": PubMed_data_module,
        "Reddit": Reddit_data_module,
        "AmazonComputers": AmazonComputers_data_module,
        "CoauthorCS": CoauthorCS_data_module,
        "TUDataset": TUDataset_data_module,
        # "ASTRO": get_astro_data_module,
    }
    
    model_dispatcher = {
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
                ),
        deterministic_train = True,
        deterministic_test = True,
        deterministic_attack = True
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
    )

    log_path = log_dir / f"{args.experiment_name}" / f"version_{trainer.logger.version}"
    
    # Train the model
    trainer.fit(model, data_module)

    # Run adversarial attacks after training
    model.adversarial_attack(
        attack_fn=targeted_class_confidence_attack,
        epsilon_list=[0.01],
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
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for training and testing"
    )
    
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=100,
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

    args = parser.parse_args()
    
    main(args)
