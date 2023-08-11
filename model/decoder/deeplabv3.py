import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))

import torch
from torch import nn
from torch.nn import functional as F
from pprint import pprint
from model.modules.attention import Attention

__all__ = ["DeepLabV3Decoder",
           "DeepLabV3PlusDecoder"
           ]


class DeepLabV3Decoder(nn.Sequential):
    def __init__(self, in_channels, out_channels=256, atrous_rates=(12, 24, 36)):
        super().__init__(
            ASPP(in_channels, out_channels, atrous_rates),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.out_channels = out_channels

    def forward(self, *features):
        return super().forward(features[-1])


class DeepLabV3PlusDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            out_channels=256,
            atrous_rates=(12, 24, 36),
            output_stride=16,
            attention_name=None,
            num_classes=1,
    ):
        super().__init__()
        if output_stride not in {8, 16}:
            raise ValueError("Output stride should be 8 or 16, got {}.".format(output_stride))

        self.out_channels = out_channels
        self.output_stride = output_stride

        self.aspp = nn.Sequential(
            ASPP(encoder_channels[-1], out_channels, atrous_rates, separable=True),
            SeparableConv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        scale_factor = 2 if output_stride == 8 else 4
        self.up = nn.UpsamplingBilinear2d(scale_factor=scale_factor)

        highres_in_channels = encoder_channels[-3]
        highres_out_channels = 48  # proposed by authors of paper
        self.block1 = nn.Sequential(
            nn.Conv2d(highres_in_channels, highres_out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(highres_out_channels),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            SeparableConv2d(
                highres_out_channels + out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.attention = Attention(name=attention_name, in_channels=out_channels)
        self.dropout = nn.Dropout2d(0.1)
        self.conv_seg = nn.Conv2d(out_channels, num_classes, 1)

    def forward(self, features):
        aspp_features = self.aspp(features[-1])
        aspp_features = self.up(aspp_features)
        high_res_features = self.block1(features[-3])
        concat_features = torch.cat([aspp_features, high_res_features], dim=1)
        fused_features = self.block2(concat_features)
        output = self.conv_seg(self.dropout(self.attention(fused_features)))

        return output


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )


class ASPPSeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__(
            SeparableConv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates, separable=False):
        super(ASPP, self).__init__()
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
        )

        rate1, rate2, rate3 = tuple(atrous_rates)
        ASPPConvModule = ASPPConv if not separable else ASPPSeparableConv

        modules.append(ASPPConvModule(in_channels, out_channels, rate1))
        modules.append(ASPPConvModule(in_channels, out_channels, rate2))
        modules.append(ASPPConvModule(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            # pprint(f"x_shape:{x.shape},x_dtype:{x.dtype}")
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class SeparableConv2d(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            bias=True,
    ):
        dephtwise_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        pointwise_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=bias,
        )
        super().__init__(dephtwise_conv, pointwise_conv)


if __name__ == "__main__":
    from model.encoder.getencoder import get_encoder
    encoder = get_encoder(name="resnet50", predicted=False, multiple_features_return=True)
    inputs = torch.rand(2, 3, 256, 256)
    outs = encoder(inputs)
    print([out.shape for out in outs])
    print(encoder.output_stride)

    decoder = DeepLabV3PlusDecoder(
        encoder_channels=encoder.out_channels,
        out_channels=256,
        attention_name='cbam',
        num_classes=1,
        atrous_rates=(12, 24, 36),
        output_stride=encoder.output_stride
    )
    outss = decoder(outs)
    print(f"outss.shape:{outss.shape}")
