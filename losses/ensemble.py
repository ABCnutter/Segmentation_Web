import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))
import torch
import torch.nn as nn
from typing import List, Optional

from losses.soft_bce import SoftBCEWithLogitsLoss
from losses.dice import DiceLoss
from losses.focal import FocalLoss
from losses.soft_ce import SoftCrossEntropyLoss


class EnsembleLoss(nn.Module):
    def __init__(self,
                 mode: str = "binary",
                 classes: Optional[List[int]] = None,
                 log_loss: bool = False,
                 from_logits: bool = True,
                 smooth: float = 0.0,
                 ignore_index_dice_focal: Optional[int] = None,
                 ignore_index_bce_ce: Optional[int] = -100,
                 eps: float = 1e-7,
                 alpha: Optional[float] = 0.25,
                 gamma: Optional[float] = 2.0,
                 normalized: bool = False,
                 reduced_threshold: Optional[float] = None,
                 weight: Optional[torch.Tensor] = None,
                 bce_smooth_factor: Optional[float] = None,
                 pos_weight: Optional[torch.Tensor] = None,
                 ce_smooth_factor: Optional[float] = None,
                 ensembleloss_weight=None,
                 reduction: str = 'mean',
                 dim: int = 1
                 ) -> None:
        super().__init__()

        if ensembleloss_weight is None:
            ensembleloss_weight = [0.6, 0.4, 0.2]
        self.mode = mode

        self.diceloss = DiceLoss(mode=mode,
                                 classes=classes,
                                 log_loss=log_loss,
                                 from_logits=from_logits,
                                 smooth=smooth,
                                 ignore_index=ignore_index_dice_focal,
                                 eps=eps,
                                 )

        self.focalloss = FocalLoss(mode=mode,
                                   alpha=alpha,
                                   gamma=gamma,
                                   reduction=reduction,
                                   normalized=normalized,
                                   ignore_index=ignore_index_dice_focal,
                                   reduced_threshold=reduced_threshold,
                                   )

        self.softbceloss = SoftBCEWithLogitsLoss(weight=weight,
                                                 ignore_index=ignore_index_bce_ce,
                                                 reduction=reduction,
                                                 smooth_factor=bce_smooth_factor,
                                                 pos_weight=pos_weight,
                                                 )
        self.softcdloss = SoftCrossEntropyLoss(reduction=reduction,
                                               smooth_factor=ce_smooth_factor,
                                               ignore_index=ignore_index_bce_ce,
                                               dim=dim,
                                               )
        self.ensembleloss_weight = ensembleloss_weight

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if self.mode == "binary":

            return self.ensembleloss_weight[0] * self.focalloss(y_pred, y_true) + self.ensembleloss_weight[1] * self.diceloss(y_pred, y_true) + self.ensembleloss_weight[2] * self.softbceloss(y_pred, y_true)

        else:
            return self.ensembleloss_weight[0] * self.focalloss(y_pred, y_true) + self.ensembleloss_weight[1] * self.diceloss(y_pred, y_true) + self.ensembleloss_weight[2] * self.softcdloss(y_pred, y_true)


if __name__ == "__main__":
    import numpy as np

    a = np.array([[1, 0, 0], [0, 0, 0], [1, 0, 1]])
    b = np.array([[1, 0, 0], [0, 0, 0], [1, 0, 1]])
    a = torch.from_numpy(a).to(torch.float32)
    b = torch.from_numpy(b).to(torch.float32)
    loss_f = EnsembleLoss()
    loss = loss_f(a, b)
    print(loss)
    print(a.shape)
    a = torch.randint(low=0, high=1, size=(12, 3, 256, 256))
    b = torch.randint(low=0, high=1, size=(12, 3, 256, 256))
