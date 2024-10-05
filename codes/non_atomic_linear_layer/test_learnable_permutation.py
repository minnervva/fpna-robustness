import torch
from linear_atomic import LearnablePermutation
import warnings

# Define the Permuter class to use the LearnablePermutation
class Permuter(torch.nn.Module):
    def __init__(self):
        super(Permuter, self).__init__()
        self.permlayer = LearnablePermutation(4)

    def forward(self, x):
        perm_matrix = self.permlayer()
        y = torch.matmul(perm_matrix, x)
        return y[0] * y[1] + y[2] * y[3]

# Instantiate and test the permutation network
permutenet = Permuter()

# Test input and target
x = torch.tensor([1., -2., 3., 4.], dtype=torch.float32)
y = torch.tensor([-10.], dtype=torch.float32)  # Known target based on best permutation

# Loss function and optimizer
loss = torch.nn.L1Loss()
optimizer = torch.optim.Adam(permutenet.parameters(), lr=0.01)

# Training loop
for i in range(1000):
    optimizer.zero_grad()
    pred = permutenet(x)
    l = loss(pred, y)
    if i % 50 == 0:
        print(f"Iteration {i}: Loss {l.item()}")
    l.backward()
    optimizer.step()

# Switch to evaluation mode
permutenet.eval()
perm_matrix = permutenet.permlayer()

print("Permutation Matrix in Evaluation Mode:")
print(perm_matrix)

# Apply the learned permutation to the input
permuted_x = torch.matmul(perm_matrix, x)
print("\nPermuted x:")
print(permuted_x)
