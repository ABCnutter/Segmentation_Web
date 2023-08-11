import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from model.modules.layers import Conv2dReLU
from model.modules.attention import Attention

__all__ = ["FCNDecoder"]


class FCNDecoder(nn.Module):
    def __init__(self, in_channels, attention_name, num_classes):
        super().__init__()
        self.attention = Attention(name=attention_name, in_channels=in_channels)
        self.conv = Conv2dReLU(in_channels, in_channels // 4, 1)
        self.cls = nn.Conv2d(in_channels // 4, num_classes, 1)

    def forward(self, x: Tensor):
        out = self.cls(self.conv(self.attention(x[-1])))
        return out


if __name__ == "__main__":
    model = FCNDecoder(in_channels=3, attention_name='cbam', num_classes=1)