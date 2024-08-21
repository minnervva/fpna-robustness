import torch

def assign_fixed_params(model):
    rng_gen = torch.Generator()
    rng_gen.manual_seed(123)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(*p.shape, generator=rng_gen, dtype=p.dtype))

# Only works with batched data would be better if it did both by default.
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
        #percent_flipped = 0.5
        #[0,1,2,3] + torch.randperm([4,5,6,7])
        indices = torch.stack([torch.stack([torch.randperm(self.in_features) for _ in range(self.out_features)]) for b in range(len(x))])
        #print(indices.shape)
        prod = torch.gather(x[:,None,:]*self.weight, dim=2, index=indices)
        return prod.sum(dim=2) + self.bias[None,:]


class AtomicLinearTest(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(AtomicLinearTest, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.randn(out_features))
    def forward(self, x):
        # return x.matmul(self.weight.t()) + self.bias
        #indices = torch.stack([torch.stack([torch.randperm(self.in_features) for _ in range(self.out_features)]) for b in range(len(x))])
        indices = torch.stack([torch.stack([torch.tensor([i for i in range(self.in_features)]) for _ in range(self.out_features)]) for b in range(len(x))])
        #print(indices.shape)
        #print((x[:,None,:]*self.weight).shape)
        prod = torch.gather(x[:,None,:]*self.weight, dim=2, index=indices)
        return prod.sum(dim=2) + self.bias[None,:]