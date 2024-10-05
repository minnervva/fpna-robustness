import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from linear_atomic import *
from tqdm import tqdm
from typing import Annotated

if __name__ == "__main__":
    
    batch_size: int = 64
    learning_rate: float = 1e-3
    num_epochs: int = 1
    temperature: Annotated[float, "0 <= temperature"] = 1.0
    device: torch.device = set_device("cuda")
    checkpoint_path: Path = Path("./non_atomic_linear_layer")
    dataset_path: Path = Path("./data")
    
    train_loader, test_loader = load_mnist(
        batch_size=batch_size,
        dataset_path=dataset_path
    )
    
    model = MNISTClassifierAtomicLinearLearnablePermutation(
        temperature=temperature
    )
    
    criterion = nn.CrossEntropyLoss()
    
    optimiser = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )
    

    # train_module_with_learnable_permutation(
    #     model=model,
    #     train_loader=train_loader,
    #     optimiser=optimiser,
    #     criterion=criterion, 
    #     device=device, 
    #     num_epochs=10
    # )
    
    checkpoint_path = checkpoint_path / f"{model.__class__.__name__}.pth"
    model = load_weights(model, checkpoint_path)
    
    # test_loop(model, train_loader, criterion, device)
    
    perm_optimiser = torch.optim.SGD(
        [param for name, param in model.named_parameters() if "permutation" in name], 
        lr=learning_rate
    )

    # maximize_loss_with_permutation(model, train_loader, perm_optimiser, criterion, device, num_epochs=10, checkpoint_path=checkpoint_path)
    # model = load_weights(model, checkpoint_path) 
    
    # test_loop(model, train_loader, criterion, device)
    
    # model = model.to(device)
    # epsilons = [1e-10 * 10**i for i in range(10)]
    # for images, labels in train_loader:
    #     for i in range(batch_size):
    #         for epsilon in epsilons:
    #             input_image = images[i].unsqueeze(0).to(device)
    #             label = labels[i].unsqueeze(0)
    #             input_image = fgsm_attack(model, input_image, label, criterion, epsilon, device)
    #             label = torch.argmax(model(input_image), dim=1) 
    #             maximize_loss_fixed_input(model, input_image, label, perm_optimiser, criterion, device, iterations=100)
    
    maximize_loss_fixed_adversarial_input(
        model=model,
        epsilons=[1e-2],
        train_loader=train_loader,
        criterion=criterion,
        device=device,
        optimiser=perm_optimiser,
        attack_fn=fgsm_attack
    )