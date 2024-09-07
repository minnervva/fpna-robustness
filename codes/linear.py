import torch
from typing import Generator, Tuple, Set, Dict, List
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

# d = 8
# n = d / torch.sqrt(torch.tensor([d * (d-1)])) * torch.tensor([d-1] + [-1] * (d-1))

# print(torch.randperm(n.size(0)))
# n = n[torch.randperm(n.size(0))]
# print(n, 1/d)
# print(torch.matmul(n, torch.tensor([1/d] * d)))

# step 1: create normal vector
# input - dimension, d
# output - (normalised) normal vector of dimension d, where the first element is distinct and the others are equal

# step 2: loop through all d possible permutations of the normal vector, there are d of them
# input - dimenson d, normal vector n
# output - permuted normal vector

# step 3: loop over example inputs, where x1 = x2 ... = xd, and find the number of unique outputs
# input - dimension d, normal vector n
# output - unique outputs

def create_normal_vector(d: int) -> torch.tensor:
    return 1 / torch.sqrt(torch.tensor([d * (d-1)])) * torch.tensor([d-1] + [-1] * (d-1))

def permuted_normal_vector(d: int, normal_vector: torch.tensor) -> Generator[torch.Tensor, None, None]:
    for i in range(d):
        # Create a copy of the original vector to avoid modifying the original vector
        permuted_vector = normal_vector.clone()
        # print(i, permuted_vector[0], permuted_vector[i])
        # Swap the 0th element with the ith element
        permuted_vector[0], permuted_vector[i] = permuted_vector[i].clone(), permuted_vector[0].clone()
        # print(i, permuted_vector[0], permuted_vector[i]) 
        # Yield the permuted vector
        yield permuted_vector

def loop_over_hyperplane(d: int, normal_vector: torch.tensor, input_tensor: torch.tensor) -> Generator[Tuple[torch.Tensor, Set[float]], None, None]:
    generator = permuted_normal_vector(d, normal_vector)
    for permuted_vector in generator:
        output = torch.matmul(permuted_vector, input_tensor)
        yield permuted_vector, output
        
def generate_output_histogram(d: int, normal_vector: torch.tensor, input_tensor: torch.tensor) -> Dict[float, List[int]]:
    generator = loop_over_hyperplane(d, normal_vector, input_tensor)
    output_dict = dict({})
    for i, (_, output) in enumerate(generator):
        output = output.item()
        if output not in output_dict.keys():
            output_dict[output] = list([i])
        else:
            output_dict[output] += [i]        
    return output_dict

def plot_histogram(output_dict: Dict, plot_path : Path = "./histogram.png", bins: int = 10) -> None:
    data = list([])
    for value, frequency_list in output_dict.items():
        data += [value] * len(frequency_list)
    
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins, color='blue', edgecolor='black', alpha=0.7)
    plt.tight_layout()
    plt.savefig(plot_path)
    
def adversarial_attack_on_hyperplane(d: int, normal_vector: torch.tensor, input_tensor: torch.tensor, attack_vector: torch.tensor, epsilon: float = 1e-12) -> Tuple[int, int]:
    generator = loop_over_hyperplane(d, normal_vector, (input_tensor + epsilon * attack_vector))
    positive, negative = 0, 0
    for i, (permuted_normal_vector, output) in enumerate(generator):
        if output >= 0: positive += 1
        elif output < 0: 
            negative += 1
            print(i)
            print(permuted_normal_vector)
    return positive, negative

def adversarial_attack_on_hyperplane(d: int, normal_vector: torch.tensor, input_tensor: torch.tensor, attack_vector: torch.tensor, epsilon: float = 1e-12) -> float:
    generator = loop_over_hyperplane(d, normal_vector, (input_tensor + epsilon * attack_vector))
    positive, negative = 0, 0
    for _, output in generator:
        if output >= 0: positive += 1
        elif output < 0: 
            negative += 1
    return positive / (positive + negative)


def plot_adversarial_attack(d: int, normal_vector: torch.tensor, input_tensor: torch.tensor, attack_vector: torch.tensor, epsilon: float = 1e-12, step_size : int = 1e2, high: int = 1e7, plot_path : Path = "./adversarial_attack.png") -> None:
    x, y = [], []
    for i in tqdm(range(0, int(high), int(step_size))):
        ratio = adversarial_attack_on_hyperplane(d, normal_vector, input_tensor, normal_vector, i * epsilon)
        x.append(i)
        y.append(ratio) 
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.tight_layout()
    plt.savefig(plot_path)
                
# Example usage:
d = 1000
normal_vector = create_normal_vector(d)
input_tensor = torch.tensor([1.0] * d) / d
# input_tensor[-1] = input_tensor[-1] - torch.tensor([-1.6e-12])

plot_adversarial_attack(d, normal_vector, input_tensor, normal_vector)

# epsilon = 1e-12
# for i in range(0, 10_000_000, 1_000):
#     print(adversarial_attack_on_hyperplane(d, normal_vector, input_tensor, normal_vector, -1 * i * epsilon))
    

# for permuted_vector in permuted_normal_vector(d, normal_vector):
#     print(permuted_vector)
#     pass

# output_dict = dict({})
# for i, (permuted_vec, output) in enumerate(loop_over_hyperplane(d, normal_vector, input_tensor)):
    
#     # print(output_dict.keys())
#     # print(output)
#     # print(output in output_dict.keys())
#     output = output.item()
#     if output not in output_dict.keys():
#         output_dict[output] = list([i])
#     else:
#         output_dict[output] += [i]
        
# print(output_dict)
# max_val, min_val = 0, 0
# for key, value in output_dict.items():
#     print(key, len(value))
#     max_val = max(key, max_val)
#     min_val = min(key, min_val)

# print(min_val, max_val)
# plot_histogram(output_dict)
        
    
#     if rslt not in outputs:
#         outputs.add(rslt.pop().item())    
# # print("Outputs Set:", outputs)
# for output in outputs:
#     print(output)

# print(0 in outputs)


       
