import sys
import os
sys.path.append(os.path.join(os.path.abspath(os.path.join(os.getcwd(), ".")), "model\modules"))
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import Conv2dReLU



class PPM(nn.Module):
    def __init__(self, in_channels, out_channels=256, pool_sizes=(6, 3, 2, 1), use_batchnorm=True, dropout=0.2):
        super().__init__()
        self.pool_sizes = pool_sizes
        out_channels_pool = int(in_channels / len(self.pool_sizes))

        self.avpool = nn.ModuleList(
            [nn.AdaptiveAvgPool2d(output_size=self.pool_sizes[i]) for i in range(len(self.pool_sizes))])

        self.cbr = Conv2dReLU(in_channels=in_channels,
                              out_channels=out_channels_pool,
                              kernel_size=1,
                              use_batchnorm=use_batchnorm)
        self.conv = Conv2dReLU(
            in_channels=in_channels * 2,
            out_channels=out_channels,
            kernel_size=1,
            use_batchnorm=use_batchnorm,
        )

        self.dropout = nn.Dropout2d(p=dropout)


    def forward(self, x):
        h, w = x.shape[-2:]
        outs = [x]
        for i in range(len(self.pool_sizes)):
            out1 = self.cbr(self.avpool[i](x))
            out = F.interpolate(out1, size=(h, w), mode='bilinear', align_corners=True)
            outs.append(out)

        outs = torch.cat(outs, dim=1)
        outs = self.conv(outs)
        output = self.dropout(outs)
        return output


if __name__ == '__main__':
    model = PPM(in_channels=2048, out_channels=128)
    # input = torch.randn(size=(32, 2048, 60, 60))
    input = torch.randn(3, 2048, 64, 64)   

    output = model(input)
    print(output.shape)
