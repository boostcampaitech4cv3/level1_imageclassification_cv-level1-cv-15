import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from config import cfg
import yaml
from loss import make_loss
from dataloader import make_dataloader

# from torch.cuda import amp

# Tensorboard 쓸거면 주석 P풀고 쓰셈

# from torch.utils.tensorboard import SummaryWriter 

from dataset import MaskBaseDataset   # dataset class import
from loss.softmax_loss import create_criterion
from model import BaseModel, ResNet34, ResNet152, EfficientNet_b7  # model.py에서 model class import
from timm.scheduler.step_lr import StepLRScheduler

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(figsize=(12, 18 + 2))  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(top=0.8)  # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = int(np.ceil(n ** 0.5))
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = "\n".join([
            f"{task} - gt: {gt_label}, pred: {pred_label}"
            for gt_label, pred_label, task
            in zip(gt_decoded_labels, pred_decoded_labels, tasks)
        ])

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure


def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


def train(data_dir, model_dir, cfg):
    seed_everything(cfg.seed)

    # model 이란 폴더 안에서 하위폴더의 path index를 매 experiment마다 늘려줌
    save_dir = increment_path(os.path.join(model_dir, cfg.name))

    # 위의 save_dir 폴더 만들기 
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    # 현재 arguments값을 config.json파일로 dump하기(나중에 hyperparameter값을 알기 위해)
    with open(os.path.join(save_dir, 'config.yml'), 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f)
    
    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    train_loader, val_loader, num_classes = make_dataloader(data_dir,cfg)
    # -- dataset
    # dataset_module = getattr(import_module("dataset"), cfg.dataset)  # default: MaskBaseDataset
    # dataset = dataset_module(
    #     data_dir=data_dir,
    # )
    # num_classes = dataset.num_classes  # 18

    # # -- augmentation
    # transform_module = getattr(import_module("dataset"), cfg.augmentation)  # default: BaseAugmentation , CustomAugmentation
    # transform = transform_module(
    #     resize=cfg.resize,
    #     cropsize = cfg.cropsize,
    #     mean=dataset.mean,
    #     std=dataset.std,
    # )
    # dataset.set_transform(transform)

    # # -- data_loader
    # train_set, val_set = dataset.split_dataset() # dataset

    # train_loader = DataLoader(
    #     train_set,
    #     batch_size=cfg.batch_size,
    #     num_workers=multiprocessing.cpu_count() // 2,
    #     shuffle=True,
    #     pin_memory=use_cuda,
    #     drop_last=True,
    # )

    # val_loader = DataLoader(
    #     val_set,
    #     batch_size=cfg.valid_batch_size,
    #     num_workers=multiprocessing.cpu_count() // 2,
    #     shuffle=False,
    #     pin_memory=use_cuda,
    #     drop_last=True,
    # )

    # -- model
    model_module = getattr(import_module("model"), cfg.model)  # default: BaseModel, ResNet34, ResNet152, EfficientNet_b7
    model = model_module(
        num_classes=num_classes
    ).to(device)
    model = torch.nn.DataParallel(model)

    # -- loss & metric
    loss_func, center_criterion = make_loss(cfg,num_classes = num_classes)
    # criterion = create_criterion(cfg.criterion)  # default: cross_entropy
    opt_module = getattr(import_module("torch.optim"), cfg.optimizer)  # SGD , Adam

    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr,
        weight_decay=5e-4
    )
    
    scheduler = StepLRScheduler(
            optimizer,
            decay_t=cfg.lr_decay_step,
            decay_rate=0.5,
            warmup_lr_init=2e-08,
            warmup_t=5,
            t_in_epochs=False,
        )

    # scheduler = StepLR(optimizer, cfg.lr_decay_step, gamma=0.5)

    # -- Tensorboard logging
    # logger = SummaryWriter(log_dir=save_dir)

    best_val_acc = 0
    best_val_loss = np.inf
    for epoch in range(cfg.epochs):
        # train loop
        model.train()
        loss_value = 0
        matches = 0
        for idx, train_batch in enumerate(train_loader):
            inputs, labels = train_batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            feat, score = model(inputs)
            preds = torch.argmax(score, dim=-1)
            #loss = criterion(outs, labels)
            loss = loss_func(score, feat, labels)
            loss.backward()
            optimizer.step()

            loss_value += loss.item()
            matches += (preds == labels).sum().item()
            if (idx + 1) % cfg.log_interval == 0:
                train_loss = loss_value / cfg.log_interval
                train_acc = matches / cfg.batch_size / cfg.log_interval
                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch}/{cfg.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                )

                # Tensorboard
                # logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                # logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)
                loss_value = 0
                matches = 0

        scheduler.step_update(epoch+1)
        if not (epoch + 1) % cfg.validation_interval : # Validation 하는 주기는 알아서 바꿔서 해도 될듯!
        
        # val loop
            with torch.no_grad():
                print("Calculating validation results...")
                model.eval()
                val_loss_items = []
                val_acc_items = []
                figure = None
                for val_batch in val_loader:
                    inputs, labels = val_batch
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outs = model(inputs)
                    preds = torch.argmax(outs, dim=-1)

                    loss_item = criterion(outs, labels).item()
                    acc_item = (labels == preds).sum().item()
                    val_loss_items.append(loss_item)
                    val_acc_items.append(acc_item)

                    if figure is None:
                        inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                        inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                        figure = grid_image(
                            inputs_np, labels, preds, n=16, shuffle=cfg.dataset != "MaskSplitByProfileDataset"
                        )

                val_loss = np.sum(val_loss_items) / len(val_loader)
                val_acc = np.sum(val_acc_items) / len(val_set)
                best_val_loss = min(best_val_loss, val_loss)
                
                if val_acc > best_val_acc:
                    print(f"New best model for val accuracy in epoch {epoch}: {val_acc:4.2%}! saving the best model..")
                    torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                    best_val_acc = val_acc
                torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
                print(
                    f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                    f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
                )
                # Tensorboard
                # logger.add_scalar("Val/loss", val_loss, epoch)
                # logger.add_scalar("Val/accuracy", val_acc, epoch)
                # logger.add_figure("results", figure, epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Mask face classification")
    parser.add_argument("--config_file",default="configs/ResNet152/config.yml",help="path to config file", type = str)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.freeze()
    print(cfg)

    data_dir = cfg.data_dir
    model_dir = cfg.model_dir
    
    train(data_dir, model_dir, cfg)
