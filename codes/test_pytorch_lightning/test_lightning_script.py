import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
import os
from tqdm import tqdm
import pandas as pd
from utils.utils import *  # Import your utility functions
from attacks.attacks import *  # Import your attack functions
from models.models import *  # Import your models
from data_and_transforms.data_and_transforms import *  # Import data


def MNIST_data_module():
    mnist_train = DataLoader(mnist_train_data, batch_size=args.batch_size)
    mnist_test = DataLoader(mnist_test_data, batch_size=args.batch_size)

    # Initialize data module
    data_module = LightningDataModule(mnist_train, mnist_test)
    return data_module


def main(args):
    # Setup data transformations and DataLoader

    dataset_dispatcher = {"MNIST": MNIST_data_module}
    model_dispatcher = {"MNISTModel": MNISTModel}

    data_module = dataset_dispatcher[args.dataset]()
    # Initialize model with a chosen epsilon for adversarial attacks
    model = LightningClassifier(model_dispatcher[args.model]())

    # Setup logger
    if args.log_dir is None:
        log_dir = "csv_logs"
    else:
        log_dir = args.log_dir
    csv_logger = pl.loggers.CSVLogger(log_dir, name=args.experiment_name)

    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=csv_logger,
        devices=args.devices,  # Use the devices argument for GPUs
    )

    # Train the model
    trainer.fit(model, data_module)

    # After training, call different attacks (can be done manually after training as well)
    log_path = os.path.join(
        log_dir, f"{args.experiment_name}/version_{trainer.logger.version}"
    )
    model.adversarial_attack(
        fgsm_attack,
        epsilon_list=[i * 1e-12 for i in range(0, int(1e3), int(1e2))],
        log_path=log_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate a model with PyTorch Lightning"
    )

    parser.add_argument(
        "--dataset", type=str, help="Dataset to classify", required=True
    )
    parser.add_argument(
        "--model", type=str, help="Model to apply to dataset", required=True
    )

    # Add arguments for batch_size, max_epochs, experiment_name, log_dir, and devices
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for training and testing"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=1, help="Number of epochs for training"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="mnist_model",
        help="Name of the experiment for logging",
    )
    parser.add_argument(
        "--log_dir", type=str, default="csv_logs", help="Directory to save logs"
    )
    parser.add_argument(
        "--devices", type=int, default=1, help="Number of GPUs to use (0 for CPU)"
    )

    args = parser.parse_args()
    main(args)
