import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from scipy.optimize import linear_sum_assignment  # Hungarian Algorithm for evaluation
from utilities import *
import random
import warnings
from tqdm import tqdm
from typing import Optional, Tuple, List, Callable
from pathlib import Path
import pandas as pd


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
        y=x.unsqueeze(dim=1) * self.weight
        #with torch.no_grad():
        indices = torch.randperm(self.in_features)
        #print(indices)
        prod = y[:,:,indices]
        return prod.sum(dim=2) + self.bias[None, :]


# Only works with batched data would be better if it did both by default.
class AtomicLinearTorch(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(AtomicLinearTorch, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.randn(out_features))

    # x has dimension [batch_size, in_features]
    def forward(self, x):
        # return x.matmul(self.weight.t()) + self.bias
        # percent_flipped = 0.5
        # [0,1,2,3] + torch.randperm([4,5,6,7])
        y=x[:, None, :] * self.weight
        with torch.no_grad():
            indices = torch.stack(
                [
                    torch.stack(
                        [torch.randperm(self.in_features) for _ in range(self.out_features)]
                    )
                    for b in range(len(x))
                ]
            )
            # print(indices.shape)
            prod = torch.gather(y, dim=2, index=indices)
        return prod.sum(dim=2) + self.bias[None, :]


class AtomicLinearTest(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(AtomicLinearTest, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        # return x.matmul(self.weight.t()) + self.bias
        # indices = torch.stack([torch.stack([torch.randperm(self.in_features) for _ in range(self.out_features)]) for b in range(len(x))])
        y = x[:, None, :] * self.weight
        with torch.no_grad():
            indices = torch.stack(
                [
                    torch.stack(
                        [
                            torch.tensor([i for i in range(self.in_features)])
                            for _ in range(self.out_features)
                        ]
                    )
                    for b in range(len(x))
                ]
            )
            # print(indices.shape)
            # print((x[:,None,:]*self.weight).shape)
            prod = torch.gather(y, dim=2, index=indices)
        return prod.sum(dim=2) + self.bias[None, :]
    

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
            # perm_matrix = torch.zeros_like(self.logits)
            # hard_perm = torch.argsort(self.logits, dim=-1)
            # perm_matrix.scatter_(1, hard_perm, 1.0)
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
    

class Classifier(torch.nn.Module):
    def __init__(self, atomics, nfeatures, nclasses, nhidden):
        super(Classifier, self).__init__()
        if atomics:
            self.layer0 = AtomicLinear(nfeatures, nhidden)
            self.layer1 = AtomicLinear(nhidden, nhidden)
            self.layer2 = AtomicLinear(nhidden, nclasses)

        else:
            self.layer0 = torch.nn.Linear(nfeatures, nhidden)
            self.layer1 = torch.nn.Linear(nhidden, nhidden)
            self.layer2 = torch.nn.Linear(nhidden, nclasses)
        
        self.norm1 = torch.nn.BatchNorm1d(nhidden)
        self.norm2 = torch.nn.BatchNorm1d(nhidden)
        self.activation = torch.nn.ReLU()
        self.probs = torch.nn.Softmax()
        assign_fixed_params(self)

    def forward(self,x):
        y0 = self.norm1(self.activation(self.layer0(x)))
        y1 = self.norm2(self.activation(self.layer1(y0)))
        y2 = self.layer2(y1)
        return y2
        #return self.probs(y1)


class ClassifierTest(torch.nn.Module):
    def __init__(self, atomics, nfeatures, nclasses, nhidden):
        super(ClassifierTest, self).__init__()
        self.layer0 = AtomicLinearTest(nfeatures, nhidden)
        self.layer1 = AtomicLinearTest(nhidden, nhidden)
        self.layer2 = AtomicLinearTest(nhidden, nclasses)
        
        self.norm1 = torch.nn.BatchNorm1d(nhidden)
        self.norm2 = torch.nn.BatchNorm1d(nhidden)
        self.activation = torch.nn.ReLU()
        self.probs = torch.nn.Softmax()
        assign_fixed_params(self)

    def forward(self,x):
        y0 = self.norm1(self.activation(self.layer0(x)))
        y1 = self.norm2(self.activation(self.layer1(y0)))
        y2 = self.layer2(y1)
        return y2
        #return self.probs(y1)


def train(model, dataloader, criterion, optimizer):
    model.train()
    for x,y in dataloader:
        optimizer.zero_grad()
        y_logits = model(x)
        _, classtype = torch.max(y,1)
        loss = criterion(y_logits, classtype)
        loss.backward()
        optimizer.step()
    #print(loss)
    #for param in model.parameters():
    #    print(param.grad)
    return loss


def test(model, dataloader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in dataloader:
            prediction_prob = model(data)
            prediction = torch.argmax(prediction_prob, dim=1)
            target = torch.argmax(target, dim=1)
            for i in range(len(prediction)):
                if prediction[i]==target[i]:
                    correct+=1
                total+=1
                
    return correct, total


def train_new_model(atomics: bool, nfeatures:int, nclasses:int, nhidden: int,train_loader, test_loader):
    model = Classifier(atomics, nfeatures, nclasses, nhidden)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    loss_history = []
    accuracy_history = []
    epochs = 25
    for epoch in range(epochs):
        loss = train(model, train_loader, criterion, optimizer).item()
        correct,total = test(model, test_loader)
        scheduler.step()
        #print(loss, correct ,total)
        loss_history.append(loss)
        accuracy_history.append(100.0*(float(correct)/float(total)))

    return model, loss_history, accuracy_history


def load_mnist(batch_size: int = 64, dataset_path: str = None) -> Tuple[DataLoader, DataLoader]:
    """Loads and Preprocesses MNIST dataset, returning DataLoaders for both the train and test set

    Args:
        batch_size (int, optional): _description_. Defaults to 64.
        dataset_path (str, optional): _description_. Defaults to None.

    Returns:
        Tuple[DataLoader, DataLoader]: _description_
    """
    # Preprocessing, Normalising and converting default PIL Images to Torch Tensors
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    
    # Load and Save MNIST dataset, splitting into the train and test set 
    train_dataset = datasets.MNIST(root=dataset_path, train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root=dataset_path, train=False, transform=transform)
    
    # Pass train and test sets into torch,utils.data DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def load_weights(model, checkpoint_path):
    """ Load checkpoint weights into model, ignoring other artifacts

    Args:
        model (_type_): _description_
        checkpoint_path (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Load the entire checkpoint
    checkpoint = torch.load(checkpoint_path)

    # Extract the model state dict from the checkpoint
    if "model_state_dict" in checkpoint:
        model_state_dict = checkpoint["model_state_dict"]
    else:
        model_state_dict = checkpoint  # If the checkpoint is a state dict itself

    # Filter out unexpected keys
    expected_keys = set(model.state_dict().keys())
    filtered_state_dict = {k: v for k, v in model_state_dict.items() if k in expected_keys}

    # Load the filtered state dict into the model
    model.load_state_dict(filtered_state_dict)

    return model

def set_device(device: str = "cuda") -> torch.device:
    """ Sets torch.device, defaulting to "cuda" 

    Args:
        device (str, optional): _description_. Defaults to "cuda".

    Returns:
        torch.device: _description_
    """
    if device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        warnings.warn("Warning: No GPU found, please note this will take a while")
    return device


def train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    optimiser: torch.optim,
    criterion: torch.nn,
    device: torch.device,
    num_epochs: int,
) -> None:
    """ Generic boilerplate training loop

    Args:
        model (nn.Module): _description_
        train_loader (DataLoader): _description_
        optimiser (torch.optim): _description_
        criterion (torch.nn): _description_
        device (torch.device): _description_
        num_epochs (int): _description_
    """
    
    for epoch in tqdm(range(num_epochs)):
        running_loss: float = 0.0
        for images, labels in tqdm(train_loader):
            
            # Send data batches to device
            images, labels = images.to(device), labels.to(device)
            optimiser.zero_grad()
            
            # Calculate model output and loss 
            logits = model(images)
            loss = criterion(logits, labels)
            
            # Optimisation step
            loss.backward() 
            optimiser.step()
            
            # Add loss to running loss
            running_loss += loss.item()
        
        # Print running_loss at each epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    
# Training loop
def train_module_with_learnable_permutation(
    model: nn.Module,
    train_loader: DataLoader,
    optimiser: torch.optim,
    criterion: torch.nn,
    device: torch.device,
    num_epochs: int=10,
    freeze_permutation: bool=True,
    checkpoint_path: Optional[Path] = Path("./non_atomic_linear_layer")
    ) -> None:
    """Train arbitrary classification model with learnable permutations

    Args:
        model (nn.Module): _description_
        train_loader (DataLoader): _description_
        optimiser (torch.optim): _description_
        criterion (torch.nn): _description_
        device (torch.device): _description_
        num_epochs (int, optional): _description_. Defaults to 10.
        freeze_permutation (bool, optional): _description_. Defaults to False.
        checkpoint_path (str, optional): _description_. Defaults to "./non_atomic_linear_layer/MNISTClassifierLearnableParameters.pth".
    """
    
    model = model.to(device)
    model.train()
    
    freeze_permutations(
        model=model,
        freeze_permutation=freeze_permutation
    )
        
    train_loop(
        model=model,
        train_loader=train_loader,
        optimiser=optimiser,
        criterion=criterion,
        device=device,
        num_epochs=num_epochs
    )
    
    checkpoint_path = checkpoint_path / f"{model.__class__.__name__}.pth"
    torch.save(model.state_dict(), checkpoint_path)


def test_loop(
    model: nn.Module, 
    test_loader: DataLoader, 
    criterion: torch.nn, 
    device: torch.device,
    return_accuracy: bool=False
    ) -> None:
    """ Test Module

    Args:
        model (nn.Module): _description_
        test_loader (DataLoader): _description_
        criterion (torch.nn): _description_
        device (torch.device): _description_
    """

    model = model.to(device)
    model.eval()
    
    correct: int = 0
    running_loss: float = 0.0
    
    with torch.no_grad(): # Don't track gradients
        for images, labels in tqdm(test_loader):
            # Send data batch to device
            images, labels = images.to(device), labels.to(device)
            
            # Calculate logits and loss
            logits = model(images)
            loss = criterion(logits, labels)
            
            # Update running loss
            running_loss += loss.item()
            
            # Calculate predictions
            pred = torch.argmax(logits, dim=1)
            
            # Calculate correct predictions
            correct += (pred == labels).sum().item()
            
    accuracy = 100 * correct / len(test_loader)
    print(f"Test Accuracy: {accuracy:.2f}%, Test Loss: {running_loss/len(test_loader):.4f}")


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

        
def maximize_loss_with_permutation(
    model: nn.Module, 
    train_loader: DataLoader, 
    optimizer: torch.optim, 
    criterion: torch.nn, 
    device: torch.device, 
    num_epochs: int=5, 
    patience: int=3, 
    checkpoint_path: Path= Path("./non_atomic_linear_layer")
    )-> None:
    """Maximise loss, backpropogating over permutation layers ONLY. Finds a single set of permutations
    which maximise loss on a given dataset, with the goal of finding a global permutation which causes
    classification flips.

    Args:
        model (nn.Module): _description_
        train_loader (DataLoader): _description_
        optimizer (torch.optim): _description_
        criterion (torch.nn): _description_
        device (torch.device): _description_
        num_epochs (int, optional): _description_. Defaults to 5.
        patience (int, optional): _description_. Defaults to 3.
        checkpoint_path (Path, optional): _description_. Defaults to Path("./non_atomic_linear_layer").
    """
    
    model.to(device)
    model.train()
    
    freeze_permutations(
        model=model, 
        freeze_permutation=True
    )

    best_loss: float = float("-inf")
    epochs_without_improvement: int = 0
    checkpoint_path: Path = checkpoint_path / f"{model.__class__.__name__}.pth"
    
    for epoch in tqdm(range(num_epochs)):
        running_loss: float = 0.0
        for images, labels in tqdm(train_loader):
            
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)
            loss = -loss # Use negative loss since we want to maximise loss

            loss.backward()
            optimizer.step()
            running_loss += -loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Maximized Loss: {epoch_loss:.4f}")

        if epoch_loss > best_loss:
            print(f"Loss increased from {best_loss:.4f} to {epoch_loss:.4f}. Saving checkpoint...")
            best_loss = epoch_loss
            epochs_without_improvement = 0

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss
            }, checkpoint_path)

        else:
            epochs_without_improvement += 1
            print(f"No increase in loss for {epochs_without_improvement} epochs.")

        if epochs_without_improvement >= patience:
            print(f"Stopping early after {patience} epochs of no increase.")
            break
        
        
def maximize_loss_fixed_input(
    model: nn.Module, 
    input_image: torch.Tensor,
    label: torch.Tensor, 
    optimizer: torch.optim,
    criterion: torch.nn,
    device: torch.device, 
    iterations: int = 10
    ) -> int:
    """ Maximise loss with respect to a fixed input. Finds a set of permutations foer each input in the dataset
    to maximise loss, with the goal of flipping a classification.

    Args:
        model (nn.Module): _description_
        input_image (torch.Tensor): _description_
        label (torch.Tensor): _description_
        optimizer (torch.optim): _description_
        criterion (torch.nn): _description_
        device (torch.device): _description_
        num_iterations (int, optional): _description_. Defaults to 1_000.
    """
    
    model.to(device)
    model.train()
    
    freeze_permutations(model, False)
    
    input_image, label = input_image.to(device), label.to(device)
    
    for iteration in range(iterations):
        optimizer.zero_grad()
        
        logits = model(input_image)
        loss = criterion(logits, label)
        
        loss = -loss
        loss.backward(retain_graph=True)
        
        optimizer.step()
        # running_loss = -loss.item()
        
    # print(f"Pred: {torch.argmax(logits, dim=1)}, Target: {label}")
    if label != torch.argmax(logits, dim=1):
        print("Classification Flip!")
        return 1
    
    return 0
        
def maximize_loss_fixed_adversarial_input(
    model: nn.Module,
    epsilons: List[float],
    train_loader: DataLoader,
    criterion: torch.nn,
    device: torch.device,
    optimiser: torch.optim,
    attack_fn: Callable
) -> None:
    
    model = model.to(device)
    batch_size: int = next(iter(train_loader))[0].shape[0]
    results: List = list([])
    for epsilon in tqdm(epsilons):
        baseline_correct: int = 0
        permutation_correct: int = 0
        for images, labels in tqdm(train_loader):
            for i in tqdm(range(batch_size)):
                image = images[i].unsqueeze(0).to(device)
                label = labels[i].unsqueeze(0).to(device)
                with torch.no_grad():
                    pred = torch.argmax(model(image), dim=1) 
                    baseline_correct += int(pred == label)
                
                attack_image = attack_fn(model, image, label, criterion, epsilon, device)
                
                with torch.no_grad():
                    attack_label = torch.argmax(model(attack_image), dim=1)
                    
                permutation_correct = baseline_correct - maximize_loss_fixed_input(
                    model, attack_image, attack_label, optimiser, criterion, device, iterations=100
                )
                
                results.append({
                    "epsilon": epsilon,
                    "baseline_accuracy": baseline_correct/len(train_loader),
                    "permutation_accuracy": permutation_correct/len(train_loader),
                })
                
                df = pd.DataFrame(results)
                
    df.to_csv("./results.csv", index=False)
    
def fgsm_attack(model, x, y, criterion, epsilon, device):
    
    x = x.to(device)
    y = y.to(device)
    model.to(device)
    
    # Make sure x requires gradient
    x.requires_grad = True

    # Forward pass
    logits = model(x)
    
    # Compute the loss
    loss = criterion(logits, y)
    
    # Zero out previous gradients
    model.zero_grad()
    
    # Compute gradients of the loss w.r.t. the input image
    loss.backward()

    # Create adversarial examples using the sign of the gradients
    x_adv = x + epsilon * x.grad.sign()

    # Ensure the values are still within [0, 1]
    x_adv = torch.clamp(x_adv, 0, 1)
    
    return x_adv