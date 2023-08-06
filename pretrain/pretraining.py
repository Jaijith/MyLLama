import os
import sys
import math
import glob
import time
from functools import partial
from pathlib import Path
from typing import Tuple, Optional

import lightning as L
from lightning.fabric.strategies import FSDPStrategy

import torch
from torch.utils.data import DataLoader
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

import numpy as np
from data.data_iterator import PackedDataset, CombinedDataset
from data.data_config import data_config

from pretrain.config import (n_chunks, train_data_dir, 
                             learning_rate, batch_size, 
                             beta1, beta2, micro_batch_size, 
                             weight_decay, max_iters, 
                             grad_clip, decay_lr, min_lr, 
                             lr_decay_iters, warmup_iters, out_dir, 
                             eval_interval, save_interval,log_interval,
                             eval_iters, n_device, model_type)
from pretrain.utils import get_lr, save_model_checkpoint


from domi.model import Block, DOMI, DomiConfig


def main(devices: int = n_device,
         train_data_dir: Path = train_data_dir,
         val_data_dir: Optional[Path] = None) ->None:
    """
    The auto wrap policy in this script is used as part of the Fully Sharded Data
    Parallelism (FSDP) strategy. FSDP is a method for training large models that
    dont fit into GPU memory. It does this by sharding (splitting up) the model's
    params across multiple GPUs, so each GPU only needs to store a portion of the
    model.
    
    The auto_wrap_policy is a function that determines how the models layers are wrapped
    for the sharding process. In this , its defined as a partial function application
    of the transformer_Auto_wrap_policy with Block as the transformer layer class.
    This means that its a policy specifically for the transformer models, and it will 
    automatically wrap layers of type 'Block' for sharding.
    
    In this case 'auto_wrap' is a hypothetical function that aplpies the 'auto_wrap_policy'
    to the model. it goes through each layer in the model, and if the layer is of
    type 'Block', it wraps that layer with 'FSDP'. The automatic wrapping is beneficial
    as it save from having to manually wrap each layer and it makes it easier to apply diff
    policies to different type of layers. for e.g., you might have one policy for trasnformer
    layer and a different policy for other types of layer
    
    The partial function is used to create a new function that has 'Block' pre-filled as the
    argument for 'transformer_layer_cls'
    """
    auto_wrap_policy = partial(
        transformer_auto_wrap_policy, transformer_layer_cls = {Block}
    )
    strategy = FSDPStrategy(auto_wrap_policy=auto_wrap_policy, activation_checkpointing = Block)
    
    fabric = L.Fabric(
        accelerator="cuda", devices=devices, precision="bf16-mixed", strategy=strategy
    )
    fabric.launch()
    fabric.seed_everything(1337)
    
    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)
    
    config = DomiConfig.from_name(model_type)
    
    train_dataloader, val_dataloader = create_dataloaders(
        batch_size=batch_size,
        block_size=config.block_size,
        fabric=fabric,
        train_data_dir=train_data_dir,
        val_data_dir=val_data_dir,
        seed=1338
    )
    
    if val_dataloader is None:
        train_dataloader = fabric.setup_dataloaders(train_dataloader)
    else:
        train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)
    
    with fabric.device:
        torch.set_default_dtype(torch.float16)
        model = DOMI(config)
        model.apply(model._init_weights)
        torch.set_default_dtype(torch.float32)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = learning_rate,
        weight_Decay = weight_decay,
        betas=(beta1,beta2)
    )
    model, optimizer = fabric.setup(model, optimizer)
    
    process_batch_size = batch_size // devices
    gradient_accumulation_iters = process_batch_size // micro_batch_size
    
    train(fabric, model, optimizer, train_dataloader, val_dataloader, gradient_accumulation_iters, devices)
        

def train(
    fabric : L.Fabric,
    model : torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader],
    grad_accum_steps: int,
    devices: int
) -> None:
    #The function enters the loop
    for iter_num, train_data in enumerate(train_dataloader):
        step_count = 0
        step_time = 0.0
        tokens = 0
        token_sec = 0.0
        prev_t1 = time.time()
        for iter_num, train_data in enumerate(train_dataloader):
            t0 = time.time()
        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num,min_lr,learning_rate,lr_decay_iters,warmup_iters) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        input_ids = train_data[:, 0: model.config.block_size].contiguous()
        targets = train_data[:, 1: model.config.block_size + 1].contiguous()
        
        is_accumulating = (iter_num + 1) % grad_accum_steps != 0 
        
        with fabric.no_backward_sync(model, enabled = is_accumulating):
            logits = model(input_ids)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index = -1
            )
            fabric.backward(loss / grad_accum_steps)
        
        t1 = time.time()
        
        #The gradients are clipped to prevent them from becoming too large
        #The optimizer step function is aclled to update the model parameters
        """
        is_accumulating varible is used to determine whether the current iteration is a
        step where gradients should be accumuated or whether its a step where an 
        optimization update should be performed.
        
        Gradient accumuation is a technique used to effectively increase the batch size
        without increasing memory usage. Instead of updating the model params after 
        every batch, the gradients from several batches and the params are updated based
        on the accumulated gradients. this is equivalent to training large batch sizes, but
        it uses less memory because only one batch needs to in memory at a time.
        """
        if not is_accumulating:
            fabric.clip_gradients(model, optimizer, max_norm = grad_clip)
            
            optimizer.step()
            optimizer.zero_grad()
            step_count += 1
            
            t1 = time.time()
            
            if val_dataloader is not None and step_count % eval_interval == 0:
                val_loss = validate(fabric, model, val_dataloader)
                fabric.print(f"step {iter_num}: val loss {val_loss:.4f}")
                fabric.barrier()
                fabric.log_dict(
                    {"iter": iter_num, "val_loss": val_loss, "step": step_count, "lr" : lr}
                )
            
            if step_count % save_interval == 0:
                fabric.print(f"Saving checkpoint to {out_dir}")
                save_model_checkpoint(
                    fabric=fabric, model=model, file_path=os.path.join(out_dir, f"iter-{iter_num:06d}-chkpt.pth")
                )
            
        dt = t1 -t0
        tokens += micro_batch_size * model.config.block_size
        step_time += t1 - prev_t1
        prev_t1 = t1
        
        if iter_num % log_interval == 0:
            tokens_sec_str = f"{tokens / step_time: .0f}" if not is_accumulating else "-"
            fabric.log_dict({
                "iter": iter_num , "train_loss" : loss, "step" : step_count, "lr" : lr
            })
            
            fabric.print(
                f"print {iter_num}: loss {loss.item(): .4f}, time: {dt*1000:.2f}ms"
            )
        
        if not is_accumulating:
            tokens = 0
            step_time = 0.0
            
        if iter_num >  max_iters:
            break
                
@torch.no_grad()
def validate(
    fabric: L.Fabric, model: torch.nn.Module, val_dataloader: DataLoader
) -> torch.Tensor:
    fabric.print("Validating...")
    """
    Model eval is set to switchoff some layers like dropout, and batch normalization
    as they are not required for evaluation.
    """
    model.eval()
    losses =  torch.zeros(eval_iters)
    for k, val_data in enumerate(val_dataloader):
        input_ids = val_data[:, 0 : model.config.block_size].contiguous()
        targets = val_data[:, 1: model.config.block_size + 1].contiguous()
        logits = model(input_ids)
        """
        The ignore_index argument tell the cross entropy function to
        ignore token with the last index which are likely EoS or padding token 
        """
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
        )
        losses[k] = loss.item()
    out = losses.mean()
    # Return to traning mode for further traning.
    model.train()
    return out

def create_dataloader(
    batch_size: int,
    block_size: int,
    data_dir: str,
    fabric,
    shuffle: bool = True,
    seed: int = 12345,
) -> DataLoader:
    datasets = []
    for prefix, _ in data_config:
        filenames = glob.glob(os.path.join(data_dir, prefix + "*"))
        dataset =  PackedDataset(
            filenames,n_chunks= n_chunks, block_size=block_size, shuffle=shuffle,
            seed=seed,num_processes=fabric.world_size, process_rank=fabric.global_rank
        )
        datasets.append(dataset)
    if not datasets:
        raise RuntimeError(f"No data found ata {data_dir}, run preparing_data first. ")
    
    weights = [weight for _, weight in data_config]
    sum_weights = sum(weights)
    weights = [el / sum_weights for el in weights]
    
    combined_dataset = CombinedDataset(datasets=datasets, seed=seed,weights=weights)
    
    return DataLoader(combined_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

def create_dataloaders(
    batch_size: int,
    block_size: int,
    fabric,
    train_data_dir: str = train_data_dir,
    val_data_dir: Optional[str] = None,
    seed: int = 12345
) -> Tuple[DataLoader, DataLoader]:
    #Increase by one as we need the next word as well
    effective_block_size = block_size + 1
    train_dataloader = create_dataloader(
        batch_size=batch_size,
        block_size=block_size,
        data_dir=train_data_dir,
        fabric=fabric,
        shuffle=True,
        seed=seed
    )
    val_dataloader = (create_dataloader(
        batch_size=batch_size,
        block_size=block_size,
        data_dir=val_data_dir,
        fabric=fabric,
        shuffle=False,
        seed=seed
    )
    if val_data_dir
    else None)
    return train_dataloader, val_dataloader
    
    
    





