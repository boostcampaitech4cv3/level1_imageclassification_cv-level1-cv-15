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
from timm.scheduler.step_lr import StepLRScheduler
from sklearn.metrics import f1_score
from sampler import RandomIdentitySampler
# from torch.cuda import amp

from dataset import MaskBaseDataset,ImageDataset   # dataset class import
from timm.scheduler.step_lr import StepLRScheduler
from loss.softmax_loss import F1Loss
from solver.scheduler_factory import create_scheduler
from sklearn.metrics import f1_score

def make_weights(labels, nclasses):
    labels = np.array(labels) 
    weight_arr = np.zeros_like(labels) 
    
    _, counts = np.unique(labels, return_counts=True) 
    for cls in range(nclasses):
        weight_arr = np.where(labels == cls, 1/counts[cls], weight_arr) 
        # 각 클래스의의 인덱스를 산출하여 해당 클래스 개수의 역수를 확률로 할당한다.
        # 이를 통해 각 클래스의 전체 가중치를 동일하게 한다.
 
    return weight_arr


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

def custom_imshow(img):
    img = img.cpu().numpy()
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()

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


def train(data_dir, model_dir, args):
    seed_everything(args.seed)

    if 'triplet' in args.loss_type:
        triplet = True
    else:
        triplet = False

    # model 이란 폴더 안에서 하위폴더의 path index를 매 experiment마다 늘려줌
    save_dir = increment_path(os.path.join(model_dir, args.name))

    # 위의 save_dir 폴더 만들기 
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    # 현재 arguments값을 config.json파일로 dump하기(나중에 hyperparameter값을 알기 위해)
    with open(os.path.join(save_dir, 'config.yml'), 'w', encoding='utf-8') as f:
        yaml.dump(args,f)
    
    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    dataset_module = getattr(import_module("dataset"), args.dataset)  # default: MaskBaseDataset
    dataset = dataset_module(
        data_dir = data_dir
    )
    num_classes = dataset.num_classes  # 18

    # -- augmentation
    transform_module = getattr(import_module("dataset"), args.augmentation)  # default: BaseAugmentation
    transform = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    dataset.set_transform(transform)

    train_set, val_set = dataset.split_dataset() # dataset

    if triplet :
        train_set = ImageDataset(dataset.train_image_with_ID, transform) 
        
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            num_workers=multiprocessing.cpu_count() // 2,
            shuffle=False,
            sampler=RandomIdentitySampler(train_set, args),
            pin_memory=use_cuda,
            drop_last=True,
        )

    else :

        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            num_workers=multiprocessing.cpu_count() // 2,
            shuffle=False,
            # sampler=sampler,
            pin_memory=use_cuda,
            drop_last=True,
        )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        # sampler=sampler,
        pin_memory=use_cuda,
        drop_last=True,
    )
    # -- model
    model_module = getattr(import_module("model"), args.model)  # default: BaseModel
    model = model_module(
        num_classes = num_classes,
        pretrained = True,
        triplet = triplet,
    ).to(device)
    model = torch.nn.DataParallel(model)

    criterion = make_loss(args,num_classes = num_classes)
    # val_criterion = torch.nn.CrossEntropyLoss()

    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.base_lr,
        weight_decay=5e-4
    )
    if args.scheduler =='cos':
        scheduler = create_scheduler(args,optimizer)
    else:
        scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)
    
    # scheduler = StepLRScheduler(
    #         optimizer,
    #         decay_t=args.lr_decay_step,
    #         decay_rate=0.5,
    #         warmup_lr_init=2e-08,
    #         warmup_t=5,
    #         t_in_epochs=False,
    #     )


    best_val_acc = 0
    best_val_f1 = 0
    patience=0

    for epoch in range(args.epochs):
        # train loop
        model.train()
        loss_value = 0
        matches = 0
        for idx, train_batch in enumerate(train_loader):
            inputs, labels = train_batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            #v=random.randint(1,2)
            # print(labels)
            optimizer.zero_grad()
            if triplet:
                if "Attention" in args.loss_type:
                    feat,score = model(inputs)
                    loss,ce_loss,tri_loss = criterion(score,feat,labels,model.module.fc.state_dict()['weight'])
                else:
                    feat,score = model(inputs)
                    loss,ce_loss,tri_loss = criterion(score,feat,labels)

            else:
                score = model(inputs)
                loss= criterion(score,labels)

            loss.backward()
            optimizer.step()

            preds = torch.argmax(score, dim=-1)

            loss_value += loss.item()
            matches += (preds == labels).sum().item()
            if (idx + 1) % args.log_interval == 0:
                train_loss = loss_value / args.log_interval
                train_acc = matches / args.batch_size / args.log_interval
                if args.scheduler == 'cos':
                    current_lr = scheduler._get_lr(epoch+1)[0]
                else:
                    current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                )

                loss_value = 0
                matches = 0
        if args.wandb:
            if args.scheduler == 'cos':
                wandb.log({ 'Train Epoch': epoch, 
                            'Total Loss' : train_loss, 
                            'CE Loss' : ce_loss,
                            'Tri Loss' : tri_loss,
                            'Learning rate': scheduler._get_lr(epoch+1)[0],
                            'Train Acc': train_acc})
            else :
                wandb.log({ 'Train Epoch': epoch, 
                            'Total Loss' : train_loss, 
                            'CE Loss' : ce_loss,
                            'Tri Loss' : tri_loss,
                            'Learning rate': get_lr(optimizer),
                            'Train Acc': train_acc})
        # if args.scheduler=='cos':
        #     scheduler.step(epoch+1)
        # else:
        #     scheduler.step()
        scheduler.step_update(epoch + 1)
        #if not (epoch + 1) % args.validation_interval : # Validation 하는 주기는 알아서 바꿔서 해도 될듯!
        
        # val loop
        corrects=[0]*18
        totals=[0]*18

        target_list=[]
        pred_list=[]

        with torch.no_grad():

                print("Calculating validation results...")
                model.eval()
                # val_loss_items = []
                val_acc_items = []
                figure = None

                for val_batch in val_loader:
                    inputs, labels = val_batch
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    for la in labels:
                        la=la.item()
                        totals[la]+=1

                    if triplet:
                        feat,outs = model(inputs)
                        # tot_loss,ce_loss,tri_loss = criterion(outs,feat,labels)
                    else:
                        outs = model(inputs)
                        # loss = criterion(outs,labels)
                    preds = torch.argmax(outs, dim=-1)

                    pred_list.extend(preds.cpu().detach().numpy())
                    target_list.extend(labels.cpu().detach().numpy())

                    # loss_item = criterion(outs, labels).item()
                    acc_item = (labels == preds).sum().item()
                    # val_loss_items.append(loss_item)
                    val_acc_items.append(acc_item)
                    
                    val_f1=f1_score(np.array(target_list),np.array(pred_list),average='macro')

                    for (la,pr) in zip(labels,preds):
                        la=la.item()
                        pr=pr.item()
                        if la==pr:
                            corrects[la]+=1

                    if figure is None:
                        inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                        inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                        figure = grid_image(
                            inputs_np, labels, preds, n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
                        )
                for ind,(c,t) in enumerate(zip(corrects,totals)):
                    print(f'label {ind}:{c/t:4.2%}')
                    if args.wandb:
                        wandb.log({f'label {ind}': c/t}) 

                # val_loss = np.sum(val_loss_items) / len(val_loader)
                val_acc = np.sum(val_acc_items) / len(val_set)
                # best_val_loss = min(best_val_loss, val_loss)
                
                if val_f1 > best_val_f1:
                    print(f"New best model for val f1 in epoch {epoch}: {val_acc:4.2%}! saving the best model..")
                    torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                    best_val_f1 = val_f1
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                # else:
                #     patience+=1
                torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
                print(
                    # f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                    f"best acc : {best_val_acc:4.2%} ||"
                    f"f1_score: {val_f1:4.2%}"
                )            
                if args.wandb:
                      wandb.log({ 'val_acc': val_acc, 
                                'val_f1' : val_f1, 
                                })


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
        #wandb.init(project="CV_competition", entity="panda0728",config=cfg)
        wandb.init(project="Test",entity="",config=cfg)
    train(data_dir, model_dir, cfg)
