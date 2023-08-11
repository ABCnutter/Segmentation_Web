import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Tuple

from model.modules.layers import Conv2dReLU
from model.modules.ppm import PPM
from model.modules.attention import Attention
from model.encoder.getencoder import get_encoder

__all__ = ["UPerDecoder",
           ]


class UPerDecoder(nn.Module):
    """Unified Perceptual Parsing for Scene Understanding
    https://arxiv.org/abs/1807.10221
    scales: Pooling scales used in PPM module applied on the last feature
    """

    def __init__(self, in_channels, channel=128, num_classes=1, attention: str = 'cbam',
                 scales=(6, 3, 2, 1)):  # [128, 256, 512, 1024]
        super().__init__()
        # PPM Module
        self.ppm = PPM(in_channels[-1], channel, scales)

        # FPN Module
        self.fpn_in = nn.ModuleList()
        self.fpn_out = nn.ModuleList()
        self.attention_in = nn.ModuleList()
        self.attention_out = nn.ModuleList()

        for in_ch in in_channels[:-1]:  # skip the top layer [128, 256, 512]
            self.fpn_in.append(Conv2dReLU(in_ch, channel, 1))
            self.fpn_out.append(Conv2dReLU(channel, channel, 3, 1, 1))
            self.attention_in.append(Attention(name=attention, in_channels=channel))
            self.attention_out.append(Attention(name=attention, in_channels=channel))

        self.attention_end = Attention(name=attention, in_channels=len(in_channels) * channel)

        self.bottleneck = Conv2dReLU(len(in_channels) * channel, channel, 3, 1, 1)
        self.dropout = nn.Dropout2d(0.1)
        self.conv_seg = nn.Conv2d(channel, num_classes, 1)

    def forward(self, features: Tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor:
        f = self.ppm(features[-1])
        fpn_features = [f]

        for i in reversed(range(len(features) - 1)):  # [2, 1, 0]
            feature = self.attention_in[i](self.fpn_in[i](features[i]))
            f = feature + F.interpolate(f, size=feature.shape[-2:], mode='bilinear', align_corners=False)
            fpn_features.append(self.attention_out[i](self.fpn_out[i](f)))

        fpn_features.reverse()
        for i in range(1, len(features)):
            fpn_features[i] = F.interpolate(fpn_features[i], size=fpn_features[0].shape[-2:], mode='bilinear',
                                            align_corners=False)

        output = self.bottleneck(self.attention_end(torch.cat(fpn_features, dim=1)))
        output = self.conv_seg(self.dropout(output))
        return output


if __name__ == '__main__':
    inputs = torch.randn(2, 3, 512, 512)
    model = get_encoder(name="mitb3", predicted=False, multiple_features_return=True)
    outs = model(inputs)

    from pprint import pprint

    pprint([v.shape for v in outs])

    upernetmodel = UPerDecoder(model.out_channels, 128)
    y = upernetmodel(outs)
    pprint(y.shape)
