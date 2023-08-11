import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))

from schedulers.cosine_lr import CosineLRScheduler
from schedulers.poly_lr import PolyLRScheduler


def set_lr_scheduler(scheduler_name, optimizer,
                     t_initial=20,
                     lr_min=1e-5,
                     cycle_mul=1.1,
                     cycle_decay=0.9,
                     cycle_limit=20,
                     warmup_t=10,
                     warmup_lr_init=1e-6,
                     warmup_prefix=True):
    if scheduler_name == "cosine":
        return CosineLRScheduler(optimizer,
                                 t_initial=t_initial,
                                 lr_min=lr_min,
                                 cycle_mul=cycle_mul,
                                 cycle_decay=cycle_decay,
                                 cycle_limit=cycle_limit,
                                 warmup_t=warmup_t,
                                 warmup_lr_init=warmup_lr_init,
                                 warmup_prefix=warmup_prefix,
                                 t_in_epochs=True,
                                 noise_range_t=None,
                                 noise_pct=0.67,
                                 noise_std=1.0,
                                 noise_seed=42,
                                 k_decay=1.0,
                                 initialize=True, )
    elif scheduler_name == "poly":
        return PolyLRScheduler(optimizer,
                               t_initial=t_initial,
                               lr_min=lr_min,
                               cycle_mul=cycle_mul,
                               cycle_decay=cycle_decay,
                               cycle_limit=cycle_limit,
                               warmup_t=warmup_t,
                               warmup_lr_init=warmup_lr_init,
                               warmup_prefix=warmup_prefix,
                               power=0.5,
                               t_in_epochs=True,
                               noise_range_t=None,
                               noise_pct=0.67,
                               noise_std=1.0,
                               noise_seed=42,
                               k_decay=1.0,
                               initialize=True, )
    else:
        raise ValueError("Please select the optimizer in [cosine, poly]!")
