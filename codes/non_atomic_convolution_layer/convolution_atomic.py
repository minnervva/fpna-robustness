import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

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


def load_weights_from_conv2d(conv2d_layer: nn.Conv2d, linear_conv2d_layer: LinearConv2d):
    """
    Load weights from nn.Conv2d layer into LinearConv2d layer.
    """
    # Flatten Conv2d weights and load into Linear layer
    conv_weights = conv2d_layer.weight.data.view(conv2d_layer.out_channels, -1)
    conv_bias = conv2d_layer.bias.data if conv2d_layer.bias is not None else None
    
    linear_conv2d_layer.linear.weight.data = conv_weights
    if conv_bias is not None:
        linear_conv2d_layer.linear.bias.data = conv_bias


class LinearResNet18(nn.Module):
    def __init__(self, pretrained=True):
        super(LinearResNet18, self).__init__()
        # Load the original resnet18
        original_resnet = resnet18(pretrained=pretrained)
        
        # Replace the convolutional layers with LinearConv2d
        self.conv1 = LinearConv2d(3, 64, kernel_size=7, stride=2, padding=3)  # Replace Conv2d
        self.bn1 = original_resnet.bn1
        self.relu = original_resnet.relu
        self.maxpool = original_resnet.maxpool
        
        # Layers from the original ResNet
        self.layer1 = self._replace_conv_with_linear(original_resnet.layer1)
        self.layer2 = self._replace_conv_with_linear(original_resnet.layer2)
        self.layer3 = self._replace_conv_with_linear(original_resnet.layer3)
        self.layer4 = self._replace_conv_with_linear(original_resnet.layer4)
        
        self.avgpool = original_resnet.avgpool
        self.fc = original_resnet.fc
    
    def _replace_conv_with_linear(self, block):
        for name, layer in block.named_children():
            if isinstance(layer, nn.Conv2d):
                # Replace Conv2d with LinearConv2d
                new_layer = LinearConv2d(
                    layer.in_channels, 
                    layer.out_channels, 
                    kernel_size=layer.kernel_size[0],  # Assuming square kernels
                    stride=layer.stride[0], 
                    padding=layer.padding[0]
                )
                setattr(block, name, new_layer)
            elif isinstance(layer, nn.Sequential):
                self._replace_conv_with_linear(layer)
            elif isinstance(layer, nn.Module):
                self._replace_conv_with_linear(layer)
        return block
    
    def forward(self, x):
        # Forward pass similar to ResNet18
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

