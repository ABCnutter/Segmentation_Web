import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))

from torch import Tensor

import torch.nn as nn

from model.modules.attention import Attention
from model.modules.ppm import PPM


class PSPNetDecoder(nn.Module):
    def __init__(self, in_channels, out_channels=256, pool_sizes=(6, 3, 2, 1), attention_name='cbam', num_classes=1,
                 use_batchnorm=True, dropout_rate=0.2) -> None:
        super().__init__()
        self.ppm = PPM(in_channels, out_channels, pool_sizes, use_batchnorm=use_batchnorm, dropout=dropout_rate)
        self.attention = Attention(name=attention_name, in_channels=out_channels)
        self.dropout = nn.Dropout2d(0.1)
        self.conv_seg = nn.Conv2d(out_channels, num_classes, 1)

    def forward(self, features: Tensor):
        feature = self.ppm(features[-1])
        feature = self.attention(feature)
        output = self.conv_seg(self.dropout(feature))

        return output


if __name__ == "__main__":
    import torch

    inputs = torch.rand(2, 2048, 16, 16)
    inputs2 = [inputs] * 4
    model = PSPNetDecoder(in_channels=2048)
    outputs = model(inputs2)
    print(outputs.shape)
