import torch
import torch.nn as nn


def assign_fixed_params(model):
    rng_gen = torch.Generator()
    rng_gen.manual_seed(123)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(*p.shape, generator=rng_gen, dtype=p.dtype))


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
        y = x[:, None, :] * self.weight
        # with torch.no_grad():
        indices = torch.randperm(self.in_features)
        # print(indices)
        prod = y[:, :, indices]
        return prod.sum(dim=2) + self.bias[None, :]


class MNISTBase(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        batch_size, channels, width, height = x.size()

        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)

        # layer 1 (b, 1*28*28) -> (b, 128)
        x = self.layer_1(x)
        x = torch.relu(x)

        # layer 2 (b, 128) -> (b, 256)
        x = self.layer_2(x)
        x = torch.relu(x)

        # layer 3 (b, 256) -> (b, 10)
        x = self.layer_3(x)

        # probability distribution over labels
        x = torch.log_softmax(x, dim=1)

        return x


class MNISTClassifier(MNISTBase):
    def __init__(self):
        super().__init__()
        # mnist images are (1, 28, 28) (channels, width, height)
        self.layer_1 = torch.nn.Linear(28 * 28, 128)
        self.layer_2 = torch.nn.Linear(128, 256)
        self.layer_3 = torch.nn.Linear(256, 10)
        assign_fixed_params(self)


class AtomicMNISTClassifier(MNISTBase):
    def __init__(self):
        super().__init__()
        # mnist images are (1, 28, 28) (channels, width, height)
        self.layer_1 = AtomicLinear(28 * 28, 128)
        self.layer_2 = AtomicLinear(128, 256)
        self.layer_3 = AtomicLinear(256, 10)
        assign_fixed_params(self)


class AstroBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.nfeatures = 8
        self.nhidden = 5 * self.nfeatures
        self.nclasses = 3

        self.norm1 = torch.nn.BatchNorm1d(self.nhidden)
        self.norm2 = torch.nn.BatchNorm1d(self.nhidden)
        self.activation = torch.nn.ReLU()
        self.probs = torch.nn.Softmax()

    # x (b, 8)
    def forward(self, x):
        y0 = self.norm1(self.activation(self.layer0(x)))
        y1 = self.norm2(self.activation(self.layer1(y0)))
        y2 = self.layer2(y1)
        return y2


class AstroClassifier(AstroBase):
    def __init__(self):
        super().__init__()
        self.layer0 = torch.nn.Linear(self.nfeatures, self.nhidden)
        self.layer1 = torch.nn.Linear(self.nhidden, self.nhidden)
        self.layer2 = torch.nn.Linear(self.nhidden, self.nclasses)
        assign_fixed_params(self)


class AtomicAstroClassifier(AstroBase):
    def __init__(self):
        super().__init__()
        self.layer0 = AtomicLinear(self.nfeatures, self.nhidden)
        self.layer1 = AtomicLinear(self.nhidden, self.nhidden)
        self.layer2 = AtomicLinear(self.nhidden, self.nclasses)
        assign_fixed_params(self)

