import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparableConv(nn.Module):


    def __init__(self,conv_module,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super().__init__()

        self.conv1 = conv_module(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = conv_module(in_channels,out_channels,1,1,0,1,1,bias=bias)


    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x