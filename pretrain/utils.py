import math
from pathlib import Path
import numpy as np

import torch
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType

from lightning.fabric.strategies import DeepSpeedStrategy, FSDPStrategy
from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict

def save_model_checkpoint(fabric, model, file_path):
    """
    Boiler plate code for retreiving and saving the state_dict
    """
    file_path = Path(file_path)
    if isinstance(fabric.strategy, DeepSpeedStrategy):
        fabric.save(file_path, {"model": model})
        fabric.barrier()
        if fabric.global_rank == 0:
            convert_zero_checkpoint_to_fp32_state_dict(file_path, file_path.with_suffix(".pth"))
        return
    
    if isinstance(fabric.strategy, FSDPStrategy):
        save_policy = FullStateDictConfig(offload_to_cpu = (fabric.world_size > 1), rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            state_dict = model._forward_module.state_dict()
    else:
        state_dict = model.state_dict()
    if fabric.global_rank == 0:
        torch.save(state_dict, file_path)
    fabric.barrier()
        


def get_lr(it,min_lr, learning_rate, lr_decay_iters, warmup_iters):
    # 1. linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2. if it >  lr_decay_iters, return min learning rate
    if it >  lr_decay_iters:
        return min_lr
    # 3. in between, use cosine decya down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1 +  math.cos(math.pi *decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


    