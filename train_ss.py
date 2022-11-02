import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path
import wandb

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from config import cfg
import yaml
from loss import make_loss
from dataloader import make_dataloader
from meter import AverageMeter

# from torch.cuda import amp

from dataset import MaskBaseDataset   # dataset class import
from loss.softmax_loss import create_criterion
from model import BaseModel, ResNet34, ResNet152, EfficientNet_b7  # model.py에서 model class import
from timm.scheduler.step_lr import StepLRScheduler

from solver.make_optimizer import make_optimizer
from solver.scheduler_factory import create_scheduler
from sklearn.metrics import f1_score

class_acc = {i : 0 for i in range(18)}
cur_acc = {i : 0 for i in range(18)}

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def class_check(labels,preds):
    global class_acc, cur_acc
    labels = labels.cpu().numpy()
    preds = preds.cpu().numpy()
    for idx,label in enumerate(labels):
        if label == preds[idx] :
            cur_acc[label] += 1
        class_acc[label] += 1
    

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

    # -- dataset
    dataset_module = getattr(import_module("dataset"), cfg.dataset)  # default: MaskBaseDataset
    dataset = dataset_module(
        data_dir=data_dir,
    )
    num_classes = dataset.num_classes

    # -- augmentation
    transform_module = getattr(import_module("dataset"), cfg.augmentation)  # default: BaseAugmentation
    transform = transform_module(
        resize=cfg.resize,
        cropsize=cfg.cropsize,
        mean=dataset.mean,
        std=dataset.std,
    )
    dataset.set_transform(transform)
    train_set, val_set = dataset.split_dataset() # dataset

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        #sampler=sampler,
        pin_memory=use_cuda,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=32,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    # train_loader, val_loader, train_set, val_set, num_classes = make_dataloader(dataset,cfg)

    # -- model
    model_module = getattr(import_module("model"), cfg.model)  # default: BaseModel, ResNet34, ResNet152, EfficientNet_b7
    model = model_module(
        num_classes=num_classes
    ).to(device)
    model = torch.nn.DataParallel(model)

    # -- loss & metric
    # loss_func = make_loss(cfg,num_classes = num_classes)
    
    # if "f1" in cfg.sampler:
    #     val_criterion = create_criterion("f1")  # default: cross_entropy
    # else:
    #     val_criterion = create_criterion("cross_entropy")  # default: cross_entropy
    criterion = torch.nn.CrossEntropyLoss()

    # Custom optimizer
    # optimizer = make_optimizer(cfg, model, center_criterion)
    
    # Optimizer
    opt_module = getattr(import_module("torch.optim"), cfg.optimizer)  # SGD, Adam
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.base_lr,
        weight_decay=5e-4
    )

    # Custom scheduler
    # scheduler = create_scheduler(cfg,optimizer)
    scheduler = StepLR(optimizer, cfg.lr_decay_step, gamma=0.5)

    # scheduler = StepLRScheduler(
    #     optimizer,
    #     decay_t=cfg.lr_decay_step,
    #     decay_rate=0.5,
    #     warmup_lr_init=2e-08,
    #     warmup_t=5,
    #     t_in_epochs=False,
    # )

    best_val_acc = 0
    best_val_loss = np.inf
    patience=0

    for epoch in range(1, cfg.epochs+1):
        scheduler.step(epoch)
        # train loop
        model.train()
        loss_value = 0
        matches = 0
        for idx, train_batch in enumerate(train_loader):
            inputs, labels = train_batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            # print(f"{idx+1}th labels:{labels}")            
            score = model(inputs)
            # feat, score = model(inputs)
            #loss = criterion(outs, labels)
            loss = criterion(score,labels)
            # if cfg.loss_type == 'f1':
            #     loss = loss_func(score,labels)
            # else:
            #     loss, ce_loss, tri_loss = loss_func(score, feat, labels)
            loss.backward()
            optimizer.zero_grad()
            optimizer.step()

            preds = torch.argmax(score, dim=-1)

            loss_value += loss.item()
            matches += (preds == labels).sum().item()
            if (idx + 1) % cfg.log_interval == 0:
                train_loss = loss_value / cfg.log_interval
                train_acc = matches / cfg.batch_size / cfg.log_interval
                current_lr = get_lr(optimizer)
                #current_lr = scheduler._get_lr(epoch)[0]
                print(
                    f"Epoch[{epoch}/{cfg.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                )

                loss_value = 0
                matches = 0

            if cfg.wandb:
                wandb.log({ 'Train Epoch': epoch, 
                            'Total Loss' : train_loss, 
                            'CE Loss' : ce_loss,
                            'Tri Loss' : tri_loss,
                            'Learning rate': scheduler._get_lr(epoch)[0],
                            'Train Acc': train_acc})   

        scheduler.step_update(epoch+1)
        mask_total=[0,0,0]
        mask_correct=[0,0,0]

        gender_total=[0,0,0]
        gender_correct=[0,0,0]

        age_total=[0,0,0]
        age_correct=[0,0,0]

        target_list=[]
        pred_list=[]
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

                    for la in labels:
                        la=la.item()
                        mask_total[la//6]+=1
                        if la%2==0:
                            gender_total[0]+=1
                        else:
                            gender_total[1]+=1
                        if la%3==0:
                            age_total[0]+=1
                        elif la%3==1:
                            age_total[1]+=1
                        else:
                            age_total[2]+=1
                    
                    outs = model(inputs)
                    preds = torch.argmax(outs, dim=-1)

                    pred_list.extend(preds.cpu().detach().numpy())
                    target_list.extend(labels.cpu().detach().numpy())
                    loss_item = criterion(outs, labels).item()
                    acc_item = (labels == preds).sum().item()
                    val_loss_items.append(loss_item)
                    val_acc_items.append(acc_item)

                    val_f1 = f1_score(np.array(target_list),np.array(pred_list),average='macro')
                    
                    for (la,pr) in zip(labels,preds):
                        la=la.item()
                        pr=pr.item()
                        if la//6==pr//6:
                            mask_correct[la//6]+=1
                        
                        if la%2==pr%2:
                            gender_correct[la%2]+=1
                        
                        if la%3==pr%3:
                            age_correct[la%3]+=1

                        if la%3==2:
                            print(pr%3,end=' ')

                    if figure is None:
                        inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                        inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                        figure = grid_image(
                            inputs_np, labels, preds, n=16, shuffle=cfg.dataset != "MaskSplitByProfileDataset"
                        )
                        
                print(f'age<30:{age_correct[0]/age_total[0]:4.2%}')
                print(f'30<=age<60:{age_correct[1]/age_total[1]:4.2%}')
                print(f'age>=60:{age_correct[2]/age_total[2]:4.2%}')

                print(f'mask_wear:{mask_correct[0]/mask_total[0]:4.2%}')
                print(f'mask_incorrect{mask_correct[1]/mask_total[1]:4.2%}')
                print(f'mask_wrong{mask_correct[2]/mask_total[2]:4.2%}')

                print(f'male:{gender_correct[0]/gender_total[0]:4.2%}')
                print(f'female{gender_correct[1]/gender_total[1]:4.2%}')

                val_loss = np.sum(val_loss_items) / len(val_loader)
                val_acc = np.sum(val_acc_items) / len(val_set)
                best_val_loss = min(best_val_loss, val_loss)

                if val_acc > best_val_acc:
                    print(f"New best model for val accuracy in epoch {epoch}: {val_acc:4.2%}! saving the best model..")
                    torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                    best_val_acc = val_acc
                else:
                    patience += 1 
                torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
                
                print(
                    f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                    f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
                )

                if cfg.wandb : 
                    wandb.log({'Val Epoch': epoch, 'Val oss' : val_loss, 'Val Acc' : val_acc})
                    for idx in range(18):
                        if class_acc[idx] and cur_acc[idx]:
                            wandb.log({f"Class {idx} accuracy" : round((cur_acc[idx]/class_acc[idx])*100,2)})
        if patience == 5:
            break

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
    
    if cfg.wandb:
        wandb.init(project="CV_competition", entity="panda0728",config=cfg)

    train(data_dir, model_dir, cfg)
