import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor

from model.base.initialization import initialize_decoder


class BaseModel(nn.Module):
    def __init__(self, use_deep_supervision=False) -> None:
        super().__init__()
        self.use_deep_supervision = use_deep_supervision

    def initialize(self) -> None:
        initialize_decoder(self.decoder)
        initialize_decoder(self.deep_supervision_decoder)

    def forward(self, x: Tensor):
        features = self.encoder(x)

        seg_out = self.decoder(features)
        seg_out = self.activate(F.interpolate(seg_out, size=x.shape[-2:], mode="bilinear"))

        if self.use_deep_supervision:
            seg_outs = [seg_out]
            deep_supervision_seg_out = self.deep_supervision_decoder(features[:-1])
            for out in deep_supervision_seg_out:
                out_f = self.activate(F.interpolate(out, size=x.shape[-2:], mode="bilinear"))
                seg_outs.append(out_f)
            return seg_outs
        else:
            return seg_out


if __name__ == "__main__":
    key = ["a", "b", "c"]
    value = [1, 2, 3]

    from collections import OrderedDict

    dicts = OrderedDict(zip(key, value))

    a = [i for i in dicts.values()]
    print(a)
