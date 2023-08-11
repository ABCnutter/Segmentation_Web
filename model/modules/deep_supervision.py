import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))

from model.encoder.getencoder import get_encoder
import torch
from torch import nn, Tensor
from model.modules.layers import Conv2dReLU
from model.modules.attention import Attention

__all__ = ["DeepSupervisionHead"]


class DeepSupervisionBlock(nn.Sequential):
    def __init__(self, in_channels, attention_name, num_classes):
        conv = Conv2dReLU(in_channels, in_channels // 4, 1)
        attention = Attention(name=attention_name, in_channels=in_channels // 4)
        cls = nn.Conv2d(in_channels // 4, num_classes, 1)
        super().__init__(conv, attention, cls)


class DeepSupervisionHead(nn.Module):
    def __init__(self, in_channels, attention_name, num_classes: int = 1):
        super().__init__()
        self.blocks = nn.ModuleList()
        for in_channel in in_channels[::-1]:
            self.blocks.append(DeepSupervisionBlock(in_channel, attention_name, num_classes))

    def forward(self, features) -> Tensor:
        outs = []
        for index, feature in enumerate(features[::-1]):
            x = self.blocks[index](feature)
            outs.append(x)
        return outs


if __name__ == '__main__':
    # dict = {"out1": [1, 2, 3],
    #         "out2": [4, 5, 6],
    #         "out3": [7, 8, 9],
    #         }
    # print(**dict)
    # from encoder.resnet import resnet_extract
    backbone = get_encoder("resnet50", predicted=False)
    head = DeepSupervisionHead(backbone.out_channels[:-1], 'cbam', 1)
    x = torch.randn(2, 3, 512, 512)
    features = backbone(x)
    outs = head(features[:-1])
    # print(outs[1].shape)
    print([out.shape for out in outs])
    # outs_f = F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=False)
    # print(out.shape for out in outs_f)
