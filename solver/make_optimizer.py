import torch

def make_optimizer(cfg, model, center_criterion):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.base_lr
        weight_decay = cfg.weight_decay
        if "bias" in key:
            lr = cfg.base_lr * cfg.bias_lr_factor
            weight_decay = cfg.weight_decay_bias

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}] # model 의 parameter들

    if cfg.optimizer == 'SGD':
        optimizer = getattr(torch.optim, cfg.optimizer)(params, momentum=cfg.momentum)
    elif cfg.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.base_lr, weight_decay=cfg.weight_decay)
    else:
        optimizer = getattr(torch.optim, cfg.optimizer)(params)
    
    
    # center loss는 parameter가 loss를 계산할때만 사용되고, model에 포함되어있지않기 떄문에 별도의 optimizer가 필요
    return optimizer
    # optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.center_lr) #center_loss의 parameter
    # return optimizer, optimizer_center
