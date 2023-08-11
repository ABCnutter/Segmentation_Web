import torch
import glo
from enum import Enum

class TrainingStatus(Enum):
    NOT_STARTED = 1
    TRINGING = 2
    STOPPED = 3
    FINISHED = 4
    FAILED = 5


def resume_training_fn(checkpoint, model, optimizer, scheduler, scaler, device, use_amp):
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    if use_amp:
        scaler.load_state_dict(checkpoint['scaler'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

