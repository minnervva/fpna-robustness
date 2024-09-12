# Sandbox for PyTorch Lightning

PyTorch Lightning has out of the box sharding support. 

- Can we leverage this for distributed multi GPU experiments for training and inference? 
- Will sharding work for small models, which don't necessarily need sharding for inference, but can be trained with Distributed Data Parallel?

Note, this is simply a sandbox for @sanjif-shanmugavelu, **not** serious code whatsoever.


## Overview

The script trains a neural network on the MNIST dataset and evaluates it using adversarial attacks. TODO: @sanjif-shanmugavelu, add more models and generalise.

## Project Directory Structure

```plaintext
fpna_robustness/codes/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md       # Information about dataset usage
│
├── models/
│   ├── models.py       # place all models here        
│   └── README.md       
|
├── attacks/
│   ├── attacks.py      # place all attacks here        
│   └── README.md       
│
├── utils/
   ├── utils.py         # place all utils here        
   └── README.md    
```

## USAGE

Install the required packages using pip:

```bash
python -m pip install -r requirements.txt
```

You can run the script with different command-line arguments to customize the training process. Here’s the general usage:

```bash
python test_lightning_script.py --batch_size <batch_size> --max_epochs <max_epochs> --experiment_name <experiment_name> --log_dir <log_dir> --devices <devices>
```

Arguments
- batch_size: Batch size for training and testing (default: 64)
- max_epochs: Number of epochs for training (default: 10)
- experiment_name: Name of the experiment for logging (default: 'mnist_model')
- log_dir: Directory to save logs (default: 'csv_logs')
- devices: Number of GPUs to use (0 for CPU, default: 1)

## Example

To train the model with a batch size of 32, for 20 epochs, log results to my_logs directory, name the experiment my_experiment, and use 2 GPUs, you would run:

```bash
python test_lightning_script.py --batch_size 32 --max_epochs 20 --experiment_name my_experiment --log_dir my_logs --devices 2
```
