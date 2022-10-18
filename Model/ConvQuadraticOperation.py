import math
import torch
import torch.nn as nn
from torch.nn import Parameter, init
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_gpu = torch.cuda.is_available()




class ConvQuadraticOperation(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride,
                 padding,
                 bias: bool = True):
        super(ConvQuadraticOperation, self).__init__()
        self.in_features = in_channels
        self.out_features = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight_r = Parameter(torch.empty(
            (out_channels, in_channels, kernel_size)))
        self.weight_g = Parameter(torch.empty(
            (out_channels, in_channels, kernel_size)))
        self.weight_b = Parameter(torch.empty(
            (out_channels, in_channels, kernel_size)))

        if bias:
            self.bias_r = Parameter(torch.empty(out_channels))
            self.bias_g = Parameter(torch.empty(out_channels))
            self.bias_b = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        nn.init.constant_(self.weight_g, 0)
        nn.init.constant_(self.weight_b, 0)
        nn.init.constant_(self.bias_g, 1)
        nn.init.constant_(self.bias_b, 0)
        self.reset_parameters()

    def __reset_bias(self):
        if self.bias_r is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_r)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias_r, -bound, bound)

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight_r, a=math.sqrt(5))

        self.__reset_bias()

    def forward(self, x):
        out = F.conv1d(x, self.weight_r, self.bias_r, self.stride, self.padding, 1, 1)\
        * F.conv1d(x, self.weight_g, self.bias_g, self.stride, self.padding, 1, 1) \
        + F.conv1d(torch.pow(x, 2), self.weight_b, self.bias_b, self.stride, self.padding, 1, 1)
        return out



class ConvTransQuadraticOperation(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride,
                 padding,
                 bias: bool = True):
        super(ConvTransQuadraticOperation, self).__init__()
        self.in_features = in_channels
        self.out_features = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight_r = Parameter(torch.empty(
            (in_channels, out_channels, kernel_size)))
        self.weight_g = Parameter(torch.empty(
            (in_channels, out_channels, kernel_size)))
        self.weight_b = Parameter(torch.empty(
            (in_channels, out_channels, kernel_size)))

        if bias:
            self.bias_r = Parameter(torch.empty(out_channels))
            self.bias_g = Parameter(torch.empty(out_channels))
            self.bias_b = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        nn.init.constant_(self.weight_g, 0)
        nn.init.constant_(self.weight_b, 0)
        nn.init.constant_(self.bias_g, 1)
        nn.init.constant_(self.bias_b, 0)
        self.reset_parameters()

    def __reset_bias(self):
        if self.bias_r is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_r)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias_r, -bound, bound)

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight_r, a=math.sqrt(5))
        self.__reset_bias()

    def forward(self, x):
        out = F.conv_transpose1d(x, self.weight_r, self.bias_r, self.stride, self.padding, 0, 1) \
              * F.conv_transpose1d(x, self.weight_g, self.bias_g, self.stride, self.padding, 0, 1) \
              + F.conv_transpose1d(torch.pow(x, 2), self.weight_b, self.bias_b, self.stride, self.padding, 0, 1)
        return out


if __name__ == '__main__':
    a = torch.randn(20, 1, 2048)
    b = ConvQuadraticOperation(1, 16, 64, 8, 28)
    c = b(a)
    print(c.shape)
    d = ConvTransQuadraticOperation(16, 1, 64, 8, 28)
    e = d(c)
    print(e.shape)
