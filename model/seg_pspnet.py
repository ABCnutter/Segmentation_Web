import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))
import torch
from torchinfo import summary

from model.base.basemodel import BaseModel
from model.encoder.getencoder import get_encoder
from model.decoder.pspnet import PSPNetDecoder
from model.modules.deep_supervision import DeepSupervisionHead
from model.modules.activate import Activation


class Seg_PSPNet(BaseModel):
    def __init__(self,
                 encoder_name="resnet50",
                 encoder_predicted=False,
                 decoder_out_channels=256,
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

        self.decoder = PSPNetDecoder(in_channels=self.encoder.out_channels[-1],
                                     out_channels=decoder_out_channels,
                                     num_classes=num_classes,
                                     pool_sizes=(6, 3, 2, 1),
                                     attention_name=attention,
                                     use_batchnorm=True,
                                     dropout_rate=0.2,
                                     )

        self.deep_supervision_decoder = DeepSupervisionHead(in_channels=self.encoder.out_channels[:-1],
                                                            attention_name=attention,
                                                            num_classes=num_classes)

        self.activate = Activation(activation)

        self.name = "{}-pspnet".format(encoder_name)
        self.num_classes = num_classes
        self.initialize()


if __name__ == "__main__":
    model = Seg_PSPNet(encoder_name="mitb5",
                       encoder_predicted=False,
                       decoder_out_channels=128,
                       attention='cbam',
                       num_classes=1,
                       activation="sigmoid",
                       use_deep_supervision=True
                       )

    input = torch.rand(2, 3, 512, 512)

    from torchinfo import summary

    # print(model.encoder.out_channels)
    # summary(model, (2, 3, 512, 512) ,device="cpu")
    out = model(input)
    # print(model)
    # print(out[0])
    print([o.shape for o in out])
