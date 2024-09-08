import torch
from typing import Generator, Tuple, Set, Dict, List, Optional, Callable
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
        # Swap the 0th element with the ith element
        permuted_vector[0], permuted_vector[i] = permuted_vector[i].clone(), permuted_vector[0].clone()
        # Yield the permuted vector
        yield permuted_vector

def loop_over_hyperplane(d: int, normal_vector: torch.tensor, input_tensor: torch.tensor) -> Generator[Tuple[torch.Tensor, Set[float]], None, None]:
    generator = permuted_normal_vector(d, normal_vector)
    for permuted_vector in generator:
        output = torch.matmul(permuted_vector, input_tensor)
        yield output
        
def generate_output_histogram(d: int, normal_vector: torch.tensor, input_tensor: torch.tensor) -> Dict[float, List[int]]:
    generator = loop_over_hyperplane(d, normal_vector, input_tensor)
    output_dict = dict({})
    for i, output in enumerate(generator):
        output = output.item()
        if output not in output_dict.keys():
            output_dict[output] = list([i])
        else:
            output_dict[output] += list([i])        
    return output_dict

def plot_curve(plot_path: Path = "./output.png", figsize: Tuple = (10, 6), tight_layout: bool = True,  plot_function: Callable = None, xlabel: str = None, ylabel: str = None, **kwargs):
    plt.figure(figsize=(10, 6))
    plot_function(**kwargs)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    

def plot_histogram(output_dict: Dict, plot_path : Path = "./histogram.png", bins: int = 100) -> None:
    x = list([])
    for value, freq in output_dict.items():
        if isinstance(freq, list):
            x += [value] * len(freq)
        elif isinstance(freq, int):
            x += [value] * freq
        else:
            raise ValueError(f"output_dict values expected to be of type int or List[int], fot {type(freq)} instead")
    plot_curve(plot_path=plot_path, plot_function=plt.hist, **{"x": x, "bins": bins})
    
# TODO: debug rectangular output
def plot_barchart(output_dict: Dict, plot_path : Path = "./barchart.png") -> None:
    x = list(output_dict.keys())
    type_check = next(iter(output_dict.values()))
    if isinstance(type_check, list):
        y = list([len(freq_list) for freq_list in output_dict.values()])
    elif isinstance(type_check, int):
        y = list(output_dict.values())
    else:
        raise ValueError(f"output_dict values expected to be of type int or List[int], fot {type(type_check)} instead")
    plot_curve(plot_path=plot_path, plot_function=plt.bar, **{"x": x, "height": y, "width": 0.5})
   
    
def adversarial_attack_on_hyperplane(d: int, normal_vector: torch.tensor, input_tensor: torch.tensor, attack_vector: torch.tensor, bias: float = 0, epsilon: float = 1e-12) -> float:
    generator = loop_over_hyperplane(d, normal_vector, (input_tensor + epsilon * attack_vector))
    positive, negative = 0, 0
    for output in generator:
        if output >= bias: positive += 1
        elif output < bias:
            negative += 1
    return positive / (positive + negative)

    
def plot_adversarial_attack(d: int, normal_vector: torch.tensor, input_tensors: List[torch.tensor], attack_vector: torch.tensor, epsilon: float = 1e-12, step_size : int = 1e2, high: int = 1e4 * 1/3, plot_path : Path = "./adversarial_attack.png") -> None:
    x, ratio_mean, ratio_std_dev = list([]), list([]), list([])
    first_zero = False
    for i in tqdm(range(0, int(high), int(step_size))):
        ratio_list = list([])
        for input_tensor in input_tensors:
            ratio = adversarial_attack_on_hyperplane(d, normal_vector, input_tensor, normal_vector, i * epsilon)
            ratio_list.append(ratio)
        ratio_list = torch.tensor(ratio_list)    
        
        x.append(i)
        ratio_mean.append(torch.mean(ratio_list))
        ratio_std_dev.append(torch.std(ratio_list))
        
        if ratio_std_dev[-1] == 0 and not first_zero:
            print(f"First zero encountered at iteration {i}, with epsilon = {i * epsilon}")
            first_zero = True
            
    plot_curve(plot_path=plot_path, plot_function=plt.errorbar, xlabel="epsilon", ylabel="% positive", **{"x": x, "y": ratio_mean, "yerr": ratio_std_dev, "fmt": "o"})


def merge_output_histogram(d: int, normal_vector: torch.tensor, input_tensors: List[torch.tensor]) -> Dict[float, int]:
    merged_dict = dict({})
    for input_tensor in tqdm(input_tensors):
        tmp_dict = generate_output_histogram(d, normal_vector, input_tensor)
        for value, freq_list in tmp_dict.items():
            if value in merged_dict.keys():
                merged_dict[value] += len(freq_list)
            else:
                merged_dict[value] = len(freq_list)
    return merged_dict

def generate_mesh(d: int, min: float, max: float, N: int) -> torch.tensor:
    grid = [torch.linspace(min, max, steps=N) for _ in range(d)]   
    mesh = torch.stack(torch.meshgrid(*grid), dim=-1).view(-1, 2)
    return mesh

if __name__ == "__main__":
    
    # Example usage:
    d = 1000
    normal_vector = create_normal_vector(d)
    input_tensor = [torch.tensor([1.0 + test/1e6] * d) / d for test in range(1_000)]
    # input_tensor = input_tensor[:1]
    # input_tensor[-1] = input_tensor[-1] - torch.tensor([-1.6e-12])

    # output_dict = generate_output_histogram(d, normal_vector, input_tensor[0])
    # plot_histogram(output_dict)
    # plot_barchart(output_dict)
    # plot_adversarial_attack(d, normal_vector, input_tensor, normal_vector)
    output_dict = merge_output_histogram(d, normal_vector, input_tensor)
    plot_histogram(output_dict)
    plot_barchart(output_dict)
    print(min(output_dict.keys()), max(output_dict.keys()))

    # # Example usage for a 4D grid
    # dimensions = 2
    # start = -10
    # end = 10
    # num_points = 20

    # result = generate_mesh(dimensions, start, end, num_points)
    # print(result.size())
    # for tensor in result:
    #     print(tensor)
    

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


       
