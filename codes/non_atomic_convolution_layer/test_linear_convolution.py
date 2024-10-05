import pytest
import torch
import torch.nn as nn
from  convolution_atomic import *

@pytest.mark.parametrize("in_channels, out_channels, kernel_size, stride, padding", [
    (3, 16, 3, 1, 1),
    (1, 8, 5, 1, 2),
    (3, 32, 7, 2, 3),
    (3, 64, 1, 1, 0)
])
def test_linear_conv2d_vs_conv2d(in_channels, out_channels, kernel_size, stride, padding):
    batch_size = 1
    height = 32
    width = 32

    conv2d_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    linear_conv2d_layer = LinearConv2d(in_channels, out_channels, kernel_size, stride, padding)

    load_weights_from_conv2d(conv2d_layer, linear_conv2d_layer)

    x = torch.randn(batch_size, in_channels, height, width)

    output_conv2d = conv2d_layer(x)
    output_linear_conv2d = linear_conv2d_layer(x)

    assert torch.allclose(output_conv2d, output_linear_conv2d, atol=1e-6), "Outputs are not close!"

    print(f"Passed for in_channels={in_channels}, out_channels={out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding}")

def test_resnet18():
    model = LinearResNet18(pretrained=True)
    assert model(torch.randn(1, 3, 224, 224)) is not None
    
if __name__ == "__main__":
    pytest.main()