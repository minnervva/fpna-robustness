from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import os

mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

mnist_train_data = MNIST(os.getcwd(), train=True, download=True, transform=mnist_transform)
    
mnist_test_data =  MNIST(os.getcwd(), train=False, download=True, transform=mnist_transform)