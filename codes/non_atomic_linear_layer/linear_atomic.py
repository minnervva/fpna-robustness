import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utilities import *
import random
import warnings
from tqdm import tqdm


class AtomicLinear(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(AtomicLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.randn(out_features))

    # x has dimension [batch_size, in_features]
    def forward(self, x):
        # return x.matmul(self.weight.t()) + self.bias
        # percent_flipped = 0.5
        # [0,1,2,3] + torch.randperm([4,5,6,7])
        y=x.unsqueeze(dim=1) * self.weight
        #with torch.no_grad():
        indices = torch.randperm(self.in_features)
        #print(indices)
        prod = y[:,:,indices]
        return prod.sum(dim=2) + self.bias[None, :]


# Only works with batched data would be better if it did both by default.
class AtomicLinearTorch(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(AtomicLinearTorch, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.randn(out_features))

    # x has dimension [batch_size, in_features]
    def forward(self, x):
        # return x.matmul(self.weight.t()) + self.bias
        # percent_flipped = 0.5
        # [0,1,2,3] + torch.randperm([4,5,6,7])
        y=x[:, None, :] * self.weight
        with torch.no_grad():
            indices = torch.stack(
                [
                    torch.stack(
                        [torch.randperm(self.in_features) for _ in range(self.out_features)]
                    )
                    for b in range(len(x))
                ]
            )
            # print(indices.shape)
            prod = torch.gather(y, dim=2, index=indices)
        return prod.sum(dim=2) + self.bias[None, :]


class AtomicLinearTest(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(AtomicLinearTest, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        # return x.matmul(self.weight.t()) + self.bias
        # indices = torch.stack([torch.stack([torch.randperm(self.in_features) for _ in range(self.out_features)]) for b in range(len(x))])
        y = x[:, None, :] * self.weight
        with torch.no_grad():
            indices = torch.stack(
                [
                    torch.stack(
                        [
                            torch.tensor([i for i in range(self.in_features)])
                            for _ in range(self.out_features)
                        ]
                    )
                    for b in range(len(x))
                ]
            )
            # print(indices.shape)
            # print((x[:,None,:]*self.weight).shape)
            prod = torch.gather(y, dim=2, index=indices)
        return prod.sum(dim=2) + self.bias[None, :]
    

class LearnablePermutation(torch.nn.Module):
    def __init__(self, in_features, temperature=1.0):
        super(LearnablePermutation, self).__init__()
        self.in_features = in_features
        self.temperature = temperature
        self.logits = torch.nn.Parameter(torch.randn(in_features, in_features))
        self.init_identity()

    def init_identity(self):
        with torch.no_grad():
            self.logits.data = torch.eye(self.in_features)
    
    def forward(self):
        
        if self.training:
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(self.logits)))
            perm_matrix = F.softmax((self.logits + gumbel_noise) / self.temperature, dim=-1)
        else:
            perm_matrix = torch.zeros_like(self.logits)
            hard_perm = torch.argsort(self.logits, dim=-1)
            perm_matrix.scatter_(1, hard_perm, 1.0)
            
        self._validate_permutation_matrix(perm_matrix) 
        return perm_matrix
    
    def _validate_permutation_matrix(self, perm_matrix):
        if self.training:
            row_sums = perm_matrix.sum(dim=-1)
            if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5):
                warnings.warn("Warning: In training mode, some rows of the permutation matrix do not sum to 1")
        else:
            if not torch.all((perm_matrix == 0) | (perm_matrix == 1)):
                warnings.warn("Warning: In eval mode, matrix contains elements that are not 0 or 1")
            
            if not (torch.all(perm_matrix.sum(dim=-1) == 1) and torch.all(perm_matrix.sum(dim=-2) == 1)):
                warnings.warn("Warning: In eval mode, not all rows and columns have exactly one '1'")
  
    
class AtomicLinearLearnablePermutation(torch.nn.Module):
    def __init__(self, in_features, out_features, temperature=1.0):
        super(AtomicLinearLearnablePermutation, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.randn(out_features))
        self.permutation = LearnablePermutation(in_features, temperature)

    def forward(self, x):
        y = x[:, None, :] * self.weight
        perm_matrix = self.permutation()
        permuted_y = torch.matmul(y, perm_matrix)
        return permuted_y.sum(dim=2) + self.bias[None, :]
    

class MNISTClassifierAtomicLinearLearnablePermutation(torch.nn.Module):
    def __init__(self, temperature=1.0):
        super(MNISTClassifierAtomicLinearLearnablePermutation, self).__init__()
        self.fc1 = AtomicLinearLearnablePermutation(28 * 28, 256, temperature)
        self.fc2 = AtomicLinearLearnablePermutation(256, 128, temperature)      
        self.fc3 = AtomicLinearLearnablePermutation(128, 10, temperature)                                          

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) 
        return x
    

class Classifier(torch.nn.Module):
    def __init__(self, atomics, nfeatures, nclasses, nhidden):
        super(Classifier, self).__init__()
        if atomics:
            self.layer0 = AtomicLinear(nfeatures, nhidden)
            self.layer1 = AtomicLinear(nhidden, nhidden)
            self.layer2 = AtomicLinear(nhidden, nclasses)

        else:
            self.layer0 = torch.nn.Linear(nfeatures, nhidden)
            self.layer1 = torch.nn.Linear(nhidden, nhidden)
            self.layer2 = torch.nn.Linear(nhidden, nclasses)
        
        self.norm1 = torch.nn.BatchNorm1d(nhidden)
        self.norm2 = torch.nn.BatchNorm1d(nhidden)
        self.activation = torch.nn.ReLU()
        self.probs = torch.nn.Softmax()
        assign_fixed_params(self)

    def forward(self,x):
        y0 = self.norm1(self.activation(self.layer0(x)))
        y1 = self.norm2(self.activation(self.layer1(y0)))
        y2 = self.layer2(y1)
        return y2
        #return self.probs(y1)


class ClassifierTest(torch.nn.Module):
    def __init__(self, atomics, nfeatures, nclasses, nhidden):
        super(ClassifierTest, self).__init__()
        self.layer0 = AtomicLinearTest(nfeatures, nhidden)
        self.layer1 = AtomicLinearTest(nhidden, nhidden)
        self.layer2 = AtomicLinearTest(nhidden, nclasses)
        
        self.norm1 = torch.nn.BatchNorm1d(nhidden)
        self.norm2 = torch.nn.BatchNorm1d(nhidden)
        self.activation = torch.nn.ReLU()
        self.probs = torch.nn.Softmax()
        assign_fixed_params(self)

    def forward(self,x):
        y0 = self.norm1(self.activation(self.layer0(x)))
        y1 = self.norm2(self.activation(self.layer1(y0)))
        y2 = self.layer2(y1)
        return y2
        #return self.probs(y1)


def train(model, dataloader, criterion, optimizer):
    model.train()
    for x,y in dataloader:
        optimizer.zero_grad()
        y_logits = model(x)
        _, classtype = torch.max(y,1)
        loss = criterion(y_logits, classtype)
        loss.backward()
        optimizer.step()
    #print(loss)
    #for param in model.parameters():
    #    print(param.grad)
    return loss


def test(model, dataloader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in dataloader:
            prediction_prob = model(data)
            prediction = torch.argmax(prediction_prob, dim=1)
            target = torch.argmax(target, dim=1)
            for i in range(len(prediction)):
                if prediction[i]==target[i]:
                    correct+=1
                total+=1
                
    return correct, total


def train_new_model(atomics: bool, nfeatures:int, nclasses:int, nhidden: int,train_loader, test_loader):
    model = Classifier(atomics, nfeatures, nclasses, nhidden)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    loss_history = []
    accuracy_history = []
    epochs = 25
    for epoch in range(epochs):
        loss = train(model, train_loader, criterion, optimizer).item()
        correct,total = test(model, test_loader)
        scheduler.step()
        #print(loss, correct ,total)
        loss_history.append(loss)
        accuracy_history.append(100.0*(float(correct)/float(total)))

    return model, loss_history, accuracy_history

# Load the MNIST dataset
def load_mnist(batch_size=64):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def set_gpu_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        torch.device("cpu")
        warnings.warn("Warning: No GPU found, please note this will take a longgggg time")
        
# Training loop
def train_mnist(model, train_loader, optimizer, criterion, device, num_epochs=10, freeze_permutation=False):
    model = model.to(device)
    model.train()
    
    if freeze_permutation:
        for layer in [model.fc1, model.fc2, model.fc3]:
            for param in layer.permutation.parameters():
                param.requires_grad = False
    
    for epoch in tqdm(range(num_epochs)):
        running_loss = 0.0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
        
    torch.save(model.state_dict(), "./MNISTClassifierLearnableParameters.pth")

# Testing loop
def test_mnist(model, test_loader, criterion, device):
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%, Test Loss: {test_loss/len(test_loader):.4f}")


def freeze_all_but_permutation(model):
    for name, param in model.named_parameters():
        if "permutation" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True


# def maximize_loss_wrt_permutation(model, train_loader, optimizer, criterion, device, num_epochs=5):
#     model.to(device)
#     model.train()
#     freeze_all_but_permutation(model)
#     for epoch in tqdm(range(num_epochs)):
#         running_loss = 0.0
#         for images, labels in tqdm(train_loader):
#             images, labels = images.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss = -loss
#             loss.backward()
#             optimizer.step()
#             running_loss += -loss.item()
#         print(f"Epoch [{epoch+1}/{num_epochs}], Maximized Loss: {running_loss / len(train_loader):.4f}")
        
def maximize_loss_wrt_permutation(model, train_loader, optimizer, criterion, device, num_epochs=5, patience=3, checkpoint_path="/home/sshanmugavelu/fpna-robustness/codes/non_atomic_linear_layer/MNISTClassifierLearnableParameters.pth"):
    model.to(device)
    model.train()
    freeze_all_but_permutation(model)

    best_loss = float('-inf')
    epochs_without_improvement = 0

    for epoch in tqdm(range(num_epochs)):
        running_loss = 0.0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)
            loss = -loss

            loss.backward()
            optimizer.step()
            running_loss += -loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Maximized Loss: {epoch_loss:.4f}")

        if epoch_loss > best_loss:
            print(f"Loss increased from {best_loss:.4f} to {epoch_loss:.4f}. Saving checkpoint...")
            best_loss = epoch_loss
            epochs_without_improvement = 0

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss
            }, checkpoint_path)

        else:
            epochs_without_improvement += 1
            print(f"No increase in loss for {epochs_without_improvement} epochs.")

        if epochs_without_improvement >= patience:
            print(f"Stopping early after {patience} epochs of no increase.")
            break
    
    print("Loss maximisation complete.")
        
def maximize_loss_for_input(model, input_image, label, optimizer, criterion, device, num_iterations=100):
    model.to(device)
    model.train()
    freeze_all_but_permutation(model)
    input_image, label = input_image.to(device), label.to(device)
    for iteration in range(num_iterations):
        optimizer.zero_grad()
        output = model(input_image)
        loss = criterion(output, label)
        loss = -loss
        loss.backward()
        optimizer.step()
        running_loss = -loss.item()
        print(f"Iteration [{iteration+1}/{num_iterations}], Maximized Loss: {running_loss:.4f}")
    print("Maximization completed.")