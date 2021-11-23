"""
This code is taken from https://github.com/locuslab/TCN/blob/master/TCN/tcn.py
"""


import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()
"""
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class Chomp2d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp2d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size, :].contiguous()


class Chomp3d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp3d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()
"""


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding_size, dimensions, dropout=0.2):
        super(TemporalBlock, self).__init__()

        #"""
        if dimensions == 1:
            conv_function = nn.Conv2d
            padding = ((kernel_size-1) * dilation, padding_size)
            dilation = (dilation, 1)
        elif dimensions == 2:
            conv_function = nn.Conv3d
            padding = ((kernel_size-1) * dilation, padding_size, padding_size)
            dilation = (dilation, 1, 1)
    
        self.conv1 = weight_norm(conv_function(
            n_inputs, n_outputs, kernel_size, stride=stride, padding=padding,
            dilation=dilation, bias=True
        ))
        self.chomp1 = Chomp(padding[0])
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(conv_function(
            n_outputs, n_outputs, kernel_size, stride=stride, padding=padding,
            dilation=dilation, bias=True
        ))
        self.chomp2 = Chomp(padding[0])
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.dropout1,
                                 self.conv2, self.chomp2, self.dropout2)

        self.downsample = conv_function(n_inputs, n_outputs, 1, bias=True) if n_inputs != n_outputs else None
        

        """
        if dimensions == 1:
            padding = (kernel_size-1) * dilation
            self.conv1 = weight_norm(nn.Conv1d(
                n_inputs, n_outputs, kernel_size, stride=stride, padding=padding,
                dilation=dilation, bias=True
            ))
            self.chomp1 = Chomp1d(padding)
            self.dropout1 = nn.Dropout(dropout)

            self.conv2 = weight_norm(nn.Conv1d(
                n_outputs, n_outputs, kernel_size, stride=stride, padding=padding,
                dilation=dilation, bias=True
            ))
            self.chomp2 = Chomp1d(padding)
            self.dropout2 = nn.Dropout(dropout)

            self.net = nn.Sequential(self.conv1, self.chomp1, self.dropout1,
                                     self.conv2, self.chomp2, self.dropout2)

            self.downsample = nn.Conv1d(n_inputs, n_outputs, 1, bias=True) if n_inputs != n_outputs else None
        
        elif dimensions == 2:
            padding = ((kernel_size-1) * dilation, padding_size)
            dilation = (dilation, 1)
            self.conv1 = weight_norm(nn.Conv2d(
                n_inputs, n_outputs, kernel_size, stride=stride, padding=padding,
                dilation=dilation, bias=True
            ))
            self.chomp1 = Chomp2d(padding[0])
            self.dropout1 = nn.Dropout(dropout)

            self.conv2 = weight_norm(nn.Conv2d(
                n_outputs, n_outputs, kernel_size, stride=stride, padding=padding,
                dilation=dilation, bias=True
            ))
            self.chomp2 = Chomp2d(padding[0])
            self.dropout2 = nn.Dropout(dropout)

            self.net = nn.Sequential(self.conv1, self.chomp1, self.dropout1,
                                     self.conv2, self.chomp2, self.dropout2)

            self.downsample = nn.Conv2d(n_inputs, n_outputs, 1, bias=True) if n_inputs != n_outputs else None
        #"""

        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return torch.tanh(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, config):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(config.model.channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = config.model.channels[i-1]
            out_channels = config.model.channels[i]
            padding_size = config.model.kernel_size // 2
            layers += [TemporalBlock(
                in_channels, out_channels, config.model.kernel_size,
                stride=1,
                dilation=dilation_size,
                padding_size=padding_size,
                dropout=0.2,
                dimensions=len(config.model.field_size)
            )]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
