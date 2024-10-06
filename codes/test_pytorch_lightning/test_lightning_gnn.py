import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid  # CORA dataset
from torch_geometric.data import DataLoader as GeoDataLoader
from torch_geometric.data import Batch, Data
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv
import os
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import argparse
from typing import Annotated, Union


class GraphSAGE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
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


class LightningClassifierGNN(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        criterion: torch.nn = nn.CrossEntropyLoss(),
        optimiser: torch.optim = torch.optim.SGD,
        deterministic_train: bool = True,
        deterministic_test: bool = True,
        deterministic_attack: bool = True
    ) -> nn.Module:
        super(LightningClassifierGNN, self).__init__()

        self.model = model
        self.criterion = criterion
        self.optimiser = optimiser
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
        
        print(f"train_batch: {train_batch}, batch_idx: {batch_idx}")
        
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

    def adversarial_attack(self, attack_fn, epsilon_list, logger, device):
        data_loader = self.trainer.datamodule.val_dataloader()
        torch.use_deterministic_algorithms(self.deterministic_attack)

        # Set model in evaluation mode
        self.model.to(device)
        self.model.eval()

        results = []

        for epsilon in tqdm(epsilon_list):
            baseline_correct = 0
            attack_correct = 0
            for data in data_loader:
                data = data.to(device)
                logits = self.model(data.x, data.edge_index)

                # Baseline accuracy before attack
                baseline_preds = logits.argmax(dim=1)
                baseline_correct += (baseline_preds[data.val_mask] == data.y[data.val_mask]).sum().item()

                # Generate adversarial examples using the passed attack function
                data_adv = attack_fn(self.model, data, epsilon, device)
                adv_logits = self.model(data_adv.x, data_adv.edge_index)

                # Compute accuracy on adversarial examples
                adv_preds = adv_logits.argmax(dim=1)
                attack_correct += (adv_preds[data.val_mask] == data.y[data.val_mask]).sum().item()

            baseline_accuracy = baseline_correct / len(data_loader.dataset)
            attack_accuracy = attack_correct / len(data_loader.dataset)

            # Log results using the logger
            logger.log_metrics({
                f'baseline_accuracy_epsilon_{epsilon}': baseline_accuracy,
                f'attack_accuracy_epsilon_{epsilon}': attack_accuracy
            })

            results.append({
                "epsilon": epsilon,
                "baseline_accuracy": baseline_accuracy,
                "attack_accuracy": attack_accuracy
            })

        # Saving results in CSV using logger
        results_df = pd.DataFrame(results)
        csv_save_path = os.path.join(logger.log_dir, 'adversarial_attack_results.csv')
        results_df.to_csv(csv_save_path, index=False)


def fgsm_attack(model, data, epsilon, device):
    data_adv = data.clone()
    data_adv.x.requires_grad = True

    logits = model(data_adv.x.to(device), data_adv.edge_index.to(device))
    loss = nn.CrossEntropyLoss()(logits[data.train_mask], data.y[data.train_mask])
    model.zero_grad()
    loss.backward()

    data_adv.x = data_adv.x + epsilon * data_adv.x.grad.sign()
    return data_adv


class LightningDataModule(pl.LightningDataModule):
    def __init__(self, dataset, batch_size):
        super(LightningDataModule, self).__init__()
        self.dataset = dataset
        self.batch_size = batch_size

    def setup(self, stage=None):
        transform = T.NormalizeFeatures()
        self.dataset = Planetoid(root='data/Planetoid', name='Cora', transform=transform)

    def train_dataloader(self):
        return GeoDataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return GeoDataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)


def CORA_data_module(batch_size):
    dataset = Planetoid(root='data/Planetoid', name='Cora', transform=T.NormalizeFeatures())
    data_module = LightningDataModule(dataset, batch_size)
    return data_module


def main(args):
    data_module = CORA_data_module(args.batch_size)

    model = GraphSAGE(in_channels=data_module.dataset.num_node_features,
                      hidden_channels=16,
                      out_channels=data_module.dataset.num_classes)

    lightning_model = LightningClassifier(model)

    # Ensure the logging directory exists
    if args.log_dir is None:
        log_dir = "csv_logs"
    else:
        log_dir = args.log_dir

    # CSV logger for both training and adversarial attack logs
    csv_logger = pl.loggers.CSVLogger(log_dir, name=args.experiment_name)

    # Trainer setup
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=csv_logger,
        devices=args.devices,
    )

    # Train the model
    trainer.fit(lightning_model, data_module)

    # Run adversarial attacks after training
    lightning_model.adversarial_attack(
        attack_fn=fgsm_attack,
        epsilon_list=[0.01, 0.05, 0.1],
        logger=csv_logger,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate a GraphSAGE model with PyTorch Lightning"
    )

    parser.add_argument(
        "--dataset", type=str, help="Dataset to classify", required=True
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for training and testing"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=100, help="Number of epochs for training"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="graphsage_cora",
        help="Name of the experiment for logging",
    )
    parser.add_argument(
        "--log_dir", type=str, default="csv_logs", help="Directory to save logs"
    )
    parser.add_argument(
        "--devices", type=int, default=1, help="Number of GPUs to use (0 for CPU)"
    )

    args = parser.parse_args()
    main(args)
