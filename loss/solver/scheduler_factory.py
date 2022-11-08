""" Scheduler Factory
Hacked together by / Copyright 2020 Ross Wightman
"""
from .cosine_lr import CosineLRScheduler
from .lr_scheduler import WarmupMultiStepLR
import torch
def create_scheduler(cfg, optimizer):
    
    if cfg.scheduler == 'cos':
        num_epochs = cfg.epochs
        # type 1
        # lr_min = 0.01 * cfg.SOLVER.BASE_LR
        # warmup_lr_init = 0.001 * cfg.SOLVER.BASE_LR
        # type 2
        lr_min = 0.002 * cfg.base_lr
        warmup_lr_init = 0.01 * cfg.base_lr
        # type 3
        # lr_min = 0.001 * cfg.SOLVER.BASE_LR
        # warmup_lr_init = 0.01 * cfg.SOLVER.BASE_LR

        warmup_t = cfg.warmup_epochs
        noise_range = None

        lr_scheduler = CosineLRScheduler(
                optimizer,
                t_initial=num_epochs,
                lr_min=lr_min,
                t_mul= 1.,
                decay_rate=0.1,
                warmup_lr_init=warmup_lr_init,
                warmup_t=warmup_t,
                cycle_limit=1,
                t_in_epochs=True,
                noise_range_t=noise_range,
                noise_pct= 0.67,
                noise_std= 1.,
                noise_seed=42,
            )
    elif cfg.scheduler == 'step':
        steps = [60, 90, 120]
        gamma = 0.1
        warmup_factor = 0.01
        warmup_iter = 10
        warmup_method = "linear"
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60,90,120], gamma=0.1)

        # lr_scheduler = WarmupMultiStepLR(
        #         optimizer,
        #         steps,
        #         gamma,
        #         warmup_factor,
        #         warmup_iter,
        #         warmup_method,
        # )
    return lr_scheduler
