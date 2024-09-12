from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import os

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
mnist_train = DataLoader(
    MNIST(os.getcwd(), train=True, download=True, transform=transform),
    batch_size=args.batch_size
)
mnist_test = DataLoader(
    MNIST(os.getcwd(), train=False, download=True, transform=transform),
    batch_size=args.batch_size
)