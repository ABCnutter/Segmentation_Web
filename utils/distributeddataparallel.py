import os

import torch.distributed as dist


def init_process(rank, world_size, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def destroy_process():
    dist.destroy_process_group()
