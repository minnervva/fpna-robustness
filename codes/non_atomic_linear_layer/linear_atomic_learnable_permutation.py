import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from linear_atomic import *
from tqdm import tqdm

if __name__ == "__main__":
    batch_size = 64
    learning_rate = 1e-3
    num_epochs = 10
    temperature = 1.0
    device = set_gpu_device()
    
    train_loader, test_loader = load_mnist(batch_size)

    model = MNISTClassifierAtomicLinearLearnablePermutation(temperature)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_mnist(model, train_loader, optimizer, criterion, device, num_epochs=15)
    
    model.load_state_dict(torch.load("/home/sshanmugavelu/fpna-robustness/codes/non_atomic_linear_layer/MNISTClassifierLearnableParameters.pth", weights_only=True)) 

    test_mnist(model, test_loader, criterion, device)
    
    perm_optimizer = torch.optim.Adam(
        [param for name, param in model.named_parameters() if "permutation" in name], 
        lr=learning_rate
    )

    maximize_loss_wrt_permutation(model, train_loader, perm_optimizer, criterion, device, num_epochs=5)
    
    checkpoint = torch.load("/home/sshanmugavelu/fpna-robustness/codes/non_atomic_linear_layer/MNISTClassifierLearnableParameters.pth")

    # Extract only the model state dict (assuming it's under the key 'model_state_dict')
    model_state_dict = checkpoint.get('model_state_dict', checkpoint)  # Adjust if your file structure is different

    # If there are any unexpected keys, you can filter them out
    expected_keys = set(model.state_dict().keys())
    model_state_dict = {k: v for k, v in model_state_dict.items() if k in expected_keys}

    # Load the cleaned state dict into the model
    model.load_state_dict(model_state_dict)
    
    # model.load_state_dict(torch.load("/home/sshanmugavelu/fpna-robustness/codes/non_atomic_linear_layer/MNISTClassifierLearnableParameters.pth", weights_only=True)) 
    
    test_mnist(model, test_loader, criterion, device)

    images, labels = next(iter(train_loader))
    for i in range(batch_size):
        input_image = images[i].unsqueeze(0)
        label = labels[i].unsqueeze(0)
        maximize_loss_for_input(model, input_image, label, perm_optimizer, criterion, device, num_iterations=100)