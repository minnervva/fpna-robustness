#!/bin/bash

# Define arrays for models and datasets
models=("GraphSAGE" "GCN" "GAT")
datasets=("CORA" "CiteSeer" "PubMed")

# Loop through each model and dataset combination
for model in "${models[@]}"
do
  for dataset in "${datasets[@]}"
  do
    # Define experiment name as model-dataset
    experiment_name="${model}-${dataset}"

    echo "Running first command for $experiment_name..."
    # Run the first command once for each model-dataset combination
    python test_lightning_gnn.py --dataset "$dataset" --model "$model" --batch_size 32 --experiment_name "$experiment_name" --devices 1 --max_epochs 25 --deterministic_attack --deterministic_train

    # Run the second command 100 times for each model-dataset combination
    for i in {1..100}
    do
      echo "Running iteration $i for $experiment_name..."
      python test_lightning_gnn.py --dataset "$dataset" --model "$model" --batch_size 32 --experiment_name "$experiment_name" --devices 1 --max_epochs 25 --deterministic_attack
    done

    # Run the final command with deterministic_attack, deterministic_train, and deterministic_test
    echo "Running final command with deterministic_test for $experiment_name..."
    python test_lightning_gnn.py --dataset "$dataset" --model "$model" --batch_size 32 --experiment_name "$experiment_name" --devices 1 --max_epochs 25 --deterministic_attack --deterministic_train --deterministic_test

  done
done
