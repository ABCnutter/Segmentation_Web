import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))
import torch
from torchinfo import summary

from model.base.basemodel import BaseModel
from model.encoder.getencoder import get_encoder
from model.decoder.upernet import UPerDecoder
from model.modules.deep_supervision import DeepSupervisionHead
from model.modules.activate import Activation


class Seg_UperNet(BaseModel):
    def __init__(self,
                 encoder_name="resnet50",
                 encoder_predicted=False,
                 decoder_out_channels=128,
                 attention="cbam",
                 num_classes=1,
                 activation="sigmoid",
                 use_deep_supervision: bool = False
                 ) -> None:
        super().__init__(use_deep_supervision)

        self.encoder = get_encoder(name=encoder_name,
                                   predicted=encoder_predicted,
                                   multiple_features_return=True
                                   )

        self.decoder = UPerDecoder(in_channels=self.encoder.out_channels,
                                   channel=decoder_out_channels,
                                   attention=attention,
                                   num_classes=num_classes
                                   )

        self.deep_supervision_decoder = DeepSupervisionHead(in_channels=self.encoder.out_channels[:-1],
                                                            attention_name=attention,
                                                            num_classes=num_classes)

        self.activate = Activation(activation)

        self.name = "{}-upernet".format(encoder_name)
        self.num_classes = num_classes
        self.initialize()


if __name__ == "__main__":
    model = Seg_UperNet(encoder_name="hrnet_w32",
                        encoder_predicted=False,
                        decoder_out_channels=128,
                        attention='cbam',
                        num_classes=1,
                        activation="sigmoid",
                        use_deep_supervision=True
                        )

    input = torch.rand(3, 3, 512, 512)
    print(model.encoder.out_channels)
    out = model(input)
    print(model.encoder.out_channels)
    print(model)
    print(out[0])
    print([o.shape for o in out])
