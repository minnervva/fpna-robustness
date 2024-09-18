import torch
from utilities import *
import random

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
        #y=x[:, None, :] * self.weight
        y=x.unsqueeze(dim=1)*self.weight
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