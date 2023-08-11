import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))
import torch
from torchinfo import summary
from torch import nn
from typing import Optional
from model.base.basemodel import BaseModel
from model.encoder.getencoder import get_encoder
from model.modules.deep_supervision import DeepSupervisionHead
from model.modules.activate import Activation
from model.decoder.deeplabv3 import DeepLabV3PlusDecoder


class Seg_DeepLabV3Plus(BaseModel):
    def __init__(
            self,
            encoder_name: str = "resnet50",
            encoder_predicted: bool = False,
            encoder_output_stride: int = 16,
            decoder_channels: int = 256,
            decoder_atrous_rates: tuple = (12, 24, 36),
            attention="cbam",
            num_classes: int = 1,
            activation: Optional[str] = None,
            use_deep_supervision: bool = False,
    ):
        super().__init__(use_deep_supervision)

        self.encoder = get_encoder(name=encoder_name,
                                   predicted=encoder_predicted,
                                   multiple_features_return=True,
                                   )

        self.decoder = DeepLabV3PlusDecoder(
            encoder_channels=self.encoder.out_channels,
            out_channels=decoder_channels,
            num_classes=1,
            atrous_rates=decoder_atrous_rates,
            output_stride=encoder_output_stride
            # if encoder_name.startswith("default-") else self.encoder.output_stride,
        )

        self.deep_supervision_decoder = DeepSupervisionHead(in_channels=self.encoder.out_channels[:-1],
                                                            attention_name=attention,
                                                            num_classes=num_classes)

        self.activate = Activation(activation)

        self.name = "{}-deeplabv3plus".format(encoder_name)
        self.num_classes = num_classes
        self.initialize()


if __name__ == "__main__":
    model = Seg_DeepLabV3Plus(encoder_name="hrnet_w32",
                              encoder_predicted=False,
                              encoder_output_stride=16,
                              decoder_channels=256,
                              decoder_atrous_rates=(12, 24, 36),
                              num_classes=1,
                              activation="sigmoid",
                              use_deep_supervision=True,
                              )

    input = torch.rand(3, 3, 512, 512)
    print(model.encoder.out_channels)
    out = model(input)
    print(model.encoder.out_channels)
    # print(model)
    print(out[0])
    print([o.shape for o in out])
