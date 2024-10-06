import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
import os
from tqdm import tqdm
import pandas as pd
from typing import List, Callable, Optional
from models.models import freeze_permutations, fgsm_attack
from pathlib import Path

class LightningClassifier(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        criterion: torch.nn = nn.CrossEntropyLoss(),
        optimiser: torch.optim.Optimizer = torch.optim.SGD,
        perm_optimiser: torch.optim.Optimizer = None,
        deterministic_train: bool=True,
        deterministic_test: bool=True,
        deterministic_attack: bool=True
    ):
        super().__init__()
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
        train_batch: torch.Tensor,
        batch_idx: int
    ) -> float:
        
        torch.use_deterministic_algorithms(self.deterministic_train)
        
        x, y = train_batch
        freeze_permutations(self.model, freeze_permutation=True)
        
        logits = self.model.forward(x)
        loss = self.loss(logits, y)
        
        self.log("train_loss", loss)
        
        return loss

    def validation_step(
        self,
        val_batch: torch.Tensor,
        batch_idx: int
    ) -> None:
        
        torch.use_deterministic_algorithms(self.deterministic_test)
        
        x, y = val_batch
        
        logits = self.model.forward(x)
        loss = self.loss(logits, y)
        
        self.log("val_loss", loss)

    def configure_optimizers(
        self
    ) -> torch.optim:
        return self.optimiser(self.model.parameters(), lr=1e-3)

    def adversarial_attack(
        self,
        epsilon_list: List[float],
        device: torch.device,
        attack_fn: Callable,
        log_path: Path
    ) -> None:
        # print(f"Running adversarial attack using {attack_fn.__name__} with epsilon={epsilon_list}")
        
        # Create a dataloader to fetch validation data
        data_loader = self.trainer.datamodule.val_dataloader()
        torch.use_deterministic_algorithms(self.deterministic_attack)
        # Ensure the model is on the correct device
        # device = self.device
        model = self.model.to(device)
        
        # Set model in evaluation mode
        model.eval()

        attack_correct: int = 0
        results: List = list([])
        
        #TODO: be smnart about inputs
        for epsilon in tqdm(epsilon_list):
            for images, labels in tqdm(data_loader):
                # Move inputs and labels to the same device as the model
                images, labels = images.to(device), labels.to(device) 

                # Generate adversarial examples using the passed attack function
                attack_images = attack_fn(model, images, labels, self.criterion, epsilon, device)

                # Forward pass the adversarial examples to see how the model performs
                attack_logits = model(attack_images)
                baseline_logits = model(images)
                
                # Compute accuracy on adversarial examples
                attack_preds = torch.argmax(attack_logits, dim=1)
                baseline_preds = baseline_logits.argmax(dim=1)
                # print(baseline, preds)
                attack_correct += (attack_preds == baseline_preds).sum().item()

            accuracy = attack_correct / len(data_loader)
            results.append({"epsilon": epsilon, "accuracy": accuracy})
        
            # Log the adversarial attack accuracy
            # self.logger.experiment('adversarial_attack_accuracy', accuracy)
            # self.logger.experiment.add_scalar('adversarial_attack_accuracy', accuracy, epsilon)
            # self.log(f'adversarial_attack_accuracy_epsilon_{epsilon}', accuracy, on_step=False, on_epoch=True)
        df_results = pd.DataFrame(results)
    
    # Save the DataFrame as a CSV file
        df_results.to_csv(f"{log_path}/{attack_fn.__name__}.csv", index=False)
        print(f"saved to {log_path}/{attack_fn.__name__}.csv")
        
    def maximize_loss_fixed_input(
        self,
        model: nn.Module, 
        image: torch.Tensor,
        label: torch.Tensor, 
        optimizer: torch.optim,
        criterion: torch.nn,
        device: torch.device, 
        iterations: int = 10
        ) -> int:

        model.to(device)
        self.model.train()
        torch.use_deterministic_algorithms(self.deterministic_attack)
        freeze_permutations(self.model, False)
        
        image, label = image.to(device), label.to(device)
        
        for iteration in range(iterations):
            self.perm_optimiser.zero_grad()
            
            logits = model(image)
            loss = self.criterion(logits, label)
            
            loss = -loss
            loss.backward(retain_graph=True)
            
            self.perm_optimiser.step()
            # running_loss = -loss.item()
            
        # print(f"Pred: {torch.argmax(logits, dim=1)}, Target: {label}")
        if label != torch.argmax(logits, dim=1):
            print("Classification Flip!")
            return 1
        
        return 0
        
    def maximize_loss_fixed_adversarial_input(
        self,
        # model: nn.Module,
        epsilon_list: List[float],
        # train_loader: DataLoader,
        # criterion: torch.nn,
        device: torch.device,
        # optimiser: torch.optim,
        attack_fn: Callable,
        log_path: Path
    ) -> None:
        
        torch.use_deterministic_algorithms(self.deterministic_attack)
        model = self.model.to(device)
        data_loader = self.trainer.datamodule.val_dataloader()
        batch_size: int = next(iter(data_loader))[0].shape[0]
        results: List = list([])
        for epsilon in tqdm(epsilon_list):
            baseline_correct: int = 0
            permutation_correct: int = 0
            for images, labels in tqdm(data_loader):
                for i in tqdm(range(batch_size)):
                    image = images[i].unsqueeze(0).to(device)
                    label = labels[i].unsqueeze(0).to(device)
                    with torch.no_grad():
                        pred = torch.argmax(model(image), dim=1) 
                        baseline_correct += int(pred == label)
                    
                    attack_image = attack_fn(model, image, label, self.criterion, epsilon, device)
                    
                    with torch.no_grad():
                        attack_label = torch.argmax(model(attack_image), dim=1)
                        
                    permutation_correct = baseline_correct - self.maximize_loss_fixed_input(
                        self.model, attack_image, attack_label, self.perm_optimiser, self.criterion, device, iterations=10
                    )
                    
                    results.append({
                        "epsilon": epsilon,
                        "baseline_accuracy": baseline_correct/len(data_loader),
                        "permutation_accuracy": permutation_correct/len(data_loader),
                    })
                    
                    df = pd.DataFrame(results)
                    
        df.to_csv(f"{log_path}/{attack_fn.__name__}.csv", index=False)
        
        
        
class LightningDataModule(pl.LightningDataModule):
    
    def __init__(self, train_dataloader, val_dataloader):
        super(LightningDataModule, self).__init__()
        self.train = train_dataloader
        self.val = val_dataloader
        # self.test = test_dataloader
    
    def setup(self, stage):
        pass
            
    def train_dataloader(self):
        return self.train
    
    def val_dataloader(self):
        return self.val
    
    # def test_dataloader(self):
    #     return self.test
        
class LightningClassifierGNN(pl.LightningModule):
    def __init__(
        self, 
        model: nn.Module,
        criterion: Callable = None,
        optimizer_fn: Callable = None,
        deterministic_train: bool = True,
        deterministic_test: bool = True,
        deterministic_attack: bool = True
    ):
        """
        A modular PyTorch Lightning GNN classifier that can handle arbitrary GNN architectures.

        Args:
            model (nn.Module): Any GNN model.
            criterion (Callable): The loss function, defaults to CrossEntropyLoss.
            optimizer_fn (Callable): Function to create an optimizer, defaults to SGD.
            deterministic_train (bool): If True, enforce deterministic algorithms during training.
            deterministic_test (bool): If True, enforce deterministic algorithms during testing.
            deterministic_attack (bool): If True, enforce deterministic algorithms during adversarial attacks.
        """
        super().__init__()
        self.model = model
        self.criterion = criterion if criterion else nn.CrossEntropyLoss()
        self.optimizer_fn = optimizer_fn

        self.deterministic_train = deterministic_train
        self.deterministic_test = deterministic_test
        self.deterministic_attack = deterministic_attack
        
        # Optional: Optimizer for "permutation" parameters if they exist
        self.perm_optimiser = torch.optim.SGD(
            [param for name, param in model.named_parameters() if "permutation" in name],
            lr=1e-3
        ) if any("permutation" in name for name, _ in model.named_parameters()) else None