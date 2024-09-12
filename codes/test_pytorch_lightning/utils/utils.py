import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
import os
from tqdm import tqdm
import pandas as pd

class LightningClassifier(pl.LightningModule):

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        
    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.model.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.model.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def adversarial_attack(self, attack_fn, epsilon_list, log_path):
        # print(f"Running adversarial attack using {attack_fn.__name__} with epsilon={epsilon_list}")
        
        # Create a dataloader to fetch validation data
        data_loader = self.trainer.datamodule.val_dataloader()

        # Ensure the model is on the correct device
        device = self.device
        self.to(device)
        
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
                x, y = x.to(device), y.to(device) 

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
        