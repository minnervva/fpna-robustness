import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(LinearConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Calculate the number of input features for the Linear layer
        self.flattened_size = in_channels * kernel_size * kernel_size
        self.linear = nn.Linear(self.flattened_size, out_channels)
        
    def forward(self, x):
        batch_size, in_channels, height, width = x.size()
        
        # Compute output dimensions
        out_height = (height - self.kernel_size + 2 * self.padding) // self.stride + 1
        out_width = (width - self.kernel_size + 2 * self.padding) // self.stride + 1
        
        # Unfold the input tensor
        x_unf = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        
        # Apply linear transformation to unfolded tensor
        x_linear = self.linear(x_unf.transpose(1, 2))
        
        # Reshape to (batch_size, out_channels, out_height, out_width)
        x_out = x_linear.transpose(1, 2).view(batch_size, self.out_channels, out_height, out_width)
        
        return x_out


def load_weights_from_conv2d(conv2d_layer: nn.Conv2d, linear_conv2d_layer: 'LinearConv2d'):
    """
    Load weights from nn.Conv2d layer into LinearConv2d layer.
    """
    # Flatten Conv2d weights and load into Linear layer
    conv_weights = conv2d_layer.weight.data.view(conv2d_layer.out_channels, -1)
    conv_bias = conv2d_layer.bias.data if conv2d_layer.bias is not None else None
    
    linear_conv2d_layer.linear.weight.data = conv_weights
    if conv_bias is not None:
        linear_conv2d_layer.linear.bias.data = conv_bias
