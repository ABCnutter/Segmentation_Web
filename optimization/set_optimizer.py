from torch import optim
from typing import Tuple


def set_optimizer(optimizer_name: str,
                  model,
                  SGD_init_lr: float = 1e-3,
                  SGD_init_lr_scale_factor: float = 0.5,
                  SGD_weight_decay: float = 5e-4,
                  SGD_momentum: float = 0.9,
                  AdamW_init_lr: float = 1e-3,
                  AdamW_init_lr_scale_factor: float = 0.5,
                  AdamW_betas: Tuple[float, float] = (0.9, 0.999),
                  AdamW_eps: float = 1e-8,
                  AdamW_weight_decay: float = 1e-2,
                  AdamW_amsgrad: bool = False
                  ):
    if optimizer_name == "SGD":
        return optim.SGD([
            {'params': model.encoder.parameters(), 'lr': SGD_init_lr},
            {'params': model.decoder.parameters(), 'lr': SGD_init_lr * SGD_init_lr_scale_factor},
            {'params': model.deep_supervision_decoder.parameters(), 'lr': SGD_init_lr * SGD_init_lr_scale_factor},
        ], lr=SGD_init_lr, weight_decay=SGD_weight_decay, momentum=SGD_momentum)
    elif optimizer_name == "AdamW":
        return optim.AdamW([
            {'params': model.encoder.parameters(), 'lr': AdamW_init_lr},
            {'params': model.decoder.parameters(), 'lr': AdamW_init_lr * AdamW_init_lr_scale_factor},
            {'params': model.deep_supervision_decoder.parameters(), 'lr': AdamW_init_lr * AdamW_init_lr_scale_factor},
        ], lr=AdamW_init_lr, betas=AdamW_betas, eps=AdamW_eps, weight_decay=AdamW_weight_decay, amsgrad=AdamW_amsgrad)
    else:
        raise ValueError("Please select the optimizer in [SGD, AdamW]!")
