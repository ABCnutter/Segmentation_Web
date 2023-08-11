import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))
import torch

from losses.soft_bce import SoftBCEWithLogitsLoss
from losses.focal import FocalLoss
from losses.dice import DiceLoss
from losses.ensemble import EnsembleLoss
from losses.soft_ce import SoftCrossEntropyLoss


def set_loss_fn(loss_fn_name='ensembleloss',
                mode='binary',
                #################################
                loss_classes=None,
                loss_log_loss=False,
                loss_from_logits=True,
                loss_dice_smooth=0.0,
                loss_dice_ignore_index=None,
                loss_eps=1e-8,
                #################################
                loss_alpha=0.25,
                loss_gamma=2.0,
                loss_focal_ignore_index=None,
                loss_normalized=False,
                loss_reduced_threshold=None,
                #################################
                loss_weight=None,
                loss_pos_weight=torch.tensor(5.0),
                loss_bce_smooth_factor=None,
                loss_bce_ignore_index=-100,
                loss_reduction="mean",
                #################################
                loss_ce_ignore_index=-100,
                loss_ce_smooth_factor=None,
                loss_dim=1,
                #################################
                ensembleloss_weight=None
                ):
    if loss_fn_name == "ensembleloss":
        return EnsembleLoss(mode=mode,
                            classes=loss_classes,
                            log_loss=loss_log_loss,
                            from_logits=loss_from_logits,
                            smooth=loss_dice_smooth,
                            ignore_index_dice_focal=loss_dice_ignore_index,
                            eps=loss_eps,
                            alpha=loss_alpha,
                            gamma=loss_gamma,
                            ignore_index_bce_ce=loss_bce_ignore_index,
                            reduction=loss_reduction,
                            normalized=loss_normalized,
                            reduced_threshold=loss_reduced_threshold,
                            weight=loss_weight,
                            bce_smooth_factor=loss_bce_smooth_factor,
                            pos_weight=loss_pos_weight,
                            ce_smooth_factor=loss_ce_smooth_factor,
                            dim=loss_dim,
                            ensembleloss_weight=ensembleloss_weight
                            )
    elif loss_fn_name == "diceloss":
        return DiceLoss(mode=mode,
                        classes=loss_classes,
                        log_loss=loss_log_loss,
                        from_logits=loss_from_logits,
                        smooth=loss_dice_smooth,
                        ignore_index=loss_dice_ignore_index,
                        eps=loss_eps
                        )
    elif loss_fn_name == "focalloss":
        return FocalLoss(mode=mode,
                         alpha=loss_alpha,
                         gamma=loss_gamma,
                         ignore_index=loss_focal_ignore_index,
                         reduction=loss_reduction,
                         normalized=loss_normalized,
                         reduced_threshold=loss_reduced_threshold
                         )
    elif loss_fn_name == "bcewithlogitsloss":
        return SoftBCEWithLogitsLoss(weight=loss_weight,
                                     ignore_index=loss_bce_ignore_index,
                                     reduction=loss_reduction,
                                     smooth_factor=loss_bce_smooth_factor,
                                     pos_weight=loss_pos_weight
                                     )
    elif loss_fn_name == "crossentropyloss":
        assert mode != 'binary'
        return SoftCrossEntropyLoss(reduction=loss_reduction,
                                    smooth_factor=loss_ce_smooth_factor,
                                    ignore_index=loss_ce_ignore_index,
                                    dim=loss_dim)
    else:
        raise ValueError(
            "Please select the loss_function in [ensembleloss, diceloss, focalloss, bcewithlogitsloss, "
            "crossentropyloss]!")
