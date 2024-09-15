import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

# CNN Model Definition
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # First Conv Layer
        x = F.max_pool2d(x, 2)     # First Pooling
        x = F.relu(self.conv2(x))  # Second Conv Layer
        x = F.max_pool2d(x, 2)     # Second Pooling
        x = F.relu(self.conv3(x))  # Third Conv Layer
        x = F.max_pool2d(x, 2)     # Third Pooling
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))    # First FC Layer
        x = self.fc2(x)            # Output Layer
        return x

# Function to train the model
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

# Function to test the model
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # Sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({accuracy:.2f}%)\n')

# Hook to capture intermediate layers
activation = {}

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

# Function to visualize and save intermediate activations
def visualize_activation(activation, layer_name, save_path):
    act = activation[layer_name].squeeze()  # Remove batch dimension
    num_filters = act.size(0)  # Number of filters in the layer
    fig, axes = plt.subplots(1, min(8, num_filters), figsize=(20, 20))
    
    for i, ax in enumerate(axes):
        if i < num_filters:
            ax.imshow(act[i].cpu().numpy(), cmap='gray')
        ax.axis('off')
    
    plt.savefig(save_path)
    plt.close()

# Main function
def main():
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    batch_size = 64
    epochs = 5
    lr = 0.01

    # Data loading and transformations
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model, optimizer, and loss function
    model = CNNModel().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # Register hooks for capturing intermediate layers
    model.conv1.register_forward_hook(get_activation('conv1'))
    model.conv2.register_forward_hook(get_activation('conv2'))
    model.conv3.register_forward_hook(get_activation('conv3'))

    # Train the model
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

    # Save the intermediate activations after passing one test image
    sample_image, _ = test_dataset[0]
    sample_image = sample_image.unsqueeze(0).to(device)  # Add batch dimension
    model(sample_image)  # Forward pass through the model

    # Save intermediate activations as images
    os.makedirs('activations', exist_ok=True)
    visualize_activation(activation, 'conv1', 'activations/conv1.png')
    visualize_activation(activation, 'conv2', 'activations/conv2.png')
    visualize_activation(activation, 'conv3', 'activations/conv3.png')

if __name__ == '__main__':
    main()
