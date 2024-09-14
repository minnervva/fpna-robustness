import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from linear_atomic import *

# Main function to train and test the model
if __name__ == "__main__":
    # Hyperparameters
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 10
    temperature = 1.0  # Temperature for Gumbel-Softmax

    # Load dataset
    train_loader, test_loader = load_mnist(batch_size)

    # Model, loss function, and optimizer
    model = MNISTClassifierAtomicLinearLearnablePermutation(temperature)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_learnable_permutation(model, train_loader, optimizer, criterion, num_epochs)

    # Test the model
    test_learnable_permutation(model, test_loader, criterion)