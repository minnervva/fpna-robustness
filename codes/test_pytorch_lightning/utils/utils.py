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
from typing import List, Callable
from models.models import freeze_permutations, fgsm_attack
from pathlib import Path

class LightningClassifier(pl.LightningModule):
    def __init__(self, model: nn.Module, deterministic_train: bool=True, deterministic_test: bool=True, deterministic_attack: bool=True):
        super().__init__()
        self.model: nn.Module = model
        self.criterion: torch.nn = nn.CrossEntropyLoss()
        self.perm_optimiser =  torch.optim.SGD(
            [param for name, param in model.named_parameters() if "permutation" in name], 
            lr=1e-3
        )
        self.deterministic_train = deterministic_train
        self.deterministic_test = deterministic_test
        self.deterministic_attack = deterministic_attack
        
    def loss(self, logits, labels):
        return self.criterion(logits, labels)

    def training_step(self, train_batch, batch_idx):
        torch.use_deterministic_algorithms(self.deterministic_train)
        x, y = train_batch
        freeze_permutations(self.model, freeze_permutation=True)
        logits = self.model.forward(x)
        loss = self.loss(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        torch.use_deterministic_algorithms(self.deterministic_test)
        x, y = val_batch
        logits = self.model.forward(x)
        loss = self.loss(logits, y)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimiser = torch.optim.SGD(self.parameters(), lr=1e-3)
        return optimiser

    def adversarial_attack(self, attack_fn, epsilon_list, log_path):
        # print(f"Running adversarial attack using {attack_fn.__name__} with epsilon={epsilon_list}")
        
        # Create a dataloader to fetch validation data
        data_loader = self.trainer.datamodule.val_dataloader()
        torch.use_deterministic_algorithms(self.deterministic_attack)
        # Ensure the model is on the correct device
        # device = self.device
        # self.to(device)
        
        # Set model in evaluation mode
        self.eval()

        total_correct = 0
        total_samples = 0

        results = list([])
        
        #TODO: be smnart about inputs
        for epsilon in tqdm(epsilon_list):
            for batch in data_loader:
                x, y = batch
                # Move inputs and labels to the same device as the model
                # x, y = x.to(device), y.to(device) 

                # Generate adversarial examples using the passed attack function
                x_adv = attack_fn(self.model, x, y, epsilon)

                # Forward pass the adversarial examples to see how the model performs
                adv_logits = self.model.forward(x_adv)
                baseline_logits = self.model.forward(x)
                
                # Compute accuracy on adversarial examples
                preds = adv_logits.argmax(dim=1)
                baseline_preds = baseline_logits.argmax(dim=1)
                # print(baseline, preds)
                total_correct += (preds == baseline_preds).sum().item()
                total_samples += x.size(0)

            accuracy = total_correct / total_samples
            results.append({"epsilon": epsilon, "accuracy": accuracy})
        
            # Log the adversarial attack accuracy
            # self.logger.experiment('adversarial_attack_accuracy', accuracy)
            # self.logger.experiment.add_scalar('adversarial_attack_accuracy', accuracy, epsilon)
            # self.log(f'adversarial_attack_accuracy_epsilon_{epsilon}', accuracy, on_step=False, on_epoch=True)
        df_results = pd.DataFrame(results)
    
    # Save the DataFrame as a CSV file
        df_results.to_csv(f"{log_path}/{attack_fn.__name__}.csv", index=False)
        
    def maximize_loss_fixed_input(
        self,
        model: nn.Module, 
        input_image: torch.Tensor,
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
        
        input_image, label = input_image.to(device), label.to(device)
        
        for iteration in range(iterations):
            self.perm_optimiser.zero_grad()
            
            logits = model(input_image)
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
        
