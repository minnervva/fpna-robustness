from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

mnist_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

mnist_train_data = MNIST(
    os.getcwd(), train=True, download=True, transform=mnist_transform
)

mnist_test_data = MNIST(
    os.getcwd(), train=False, download=True, transform=mnist_transform
)


def one_hot(array):
    unique, inverse = np.unique(array, return_inverse=True)
    onehot = np.eye(unique.shape[0])[inverse]
    return onehot


# Creating our dataset class
class BuildData(torch.utils.data.Dataset):
    # Constructor
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.len = len(self.x)

    # Getting the data
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    # Getting length of the data
    def __len__(self):
        return self.len


def get_astro_data(train_batch=200, test_batch=1):
    data = pd.read_csv("data/star_classification.csv")  # , nrows = 2000)

    xcols = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]
    ycol = "class"
    classes = len(data[ycol].unique())

    X_train, X_test, y_train, y_test = train_test_split(
        data[xcols], data[ycol], random_state=123
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train.values)
    X_train = torch.tensor(scaler.transform(X_train.values)).to(torch.float32)
    y_train = torch.tensor(one_hot(y_train.values))
    X_test = torch.tensor(scaler.transform(X_test.values)).to(torch.float32)
    y_test = torch.tensor(one_hot(y_test.values))

    train_dataset = BuildData(X_train, y_train)
    test_dataset = BuildData(X_test, y_test)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=train_batch
    )  # , num_workers=1)#, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=test_batch
    )  # , num_workers=1)

    return len(xcols), classes, train_loader, test_loader

