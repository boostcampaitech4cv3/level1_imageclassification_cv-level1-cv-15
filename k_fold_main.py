import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path
from adamp import AdamP
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from loss import FocalLoss
import pandas as pd

from loss import F1Loss
from mixup import mixup,mixed_criterion
from timm.scheduler.step_lr import StepLRScheduler
from sklearn.metrics import f1_score

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def make_weights(labels, nclasses):
    labels = np.array(labels) 
    weight_arr = np.zeros_like(labels) 
    
    _, counts = np.unique(labels, return_counts=True) 
    for cls in range(nclasses):
        weight_arr = np.where(labels == cls, 1/counts[cls], weight_arr) 
        # 각 클래스의의 인덱스를 산출하여 해당 클래스 개수의 역수를 확률로 할당한다.
        # 이를 통해 각 클래스의 전체 가중치를 동일하게 한다.
 
    return weight_arr

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

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
    df=pd.read_csv('input/data/train/kfold.csv')
    df=df.sample(
    frac=1,
    random_state=42).reset_index()
    for i in range(5):
        save_dir = increment_path(os.path.join(model_dir,args.name))
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=4)
        
        use_cuda = torch.cuda.is_available()
        device=torch.device('cuda' if use_cuda else 'cpu')

        dataset_module=getattr(import_module('dataset'),args.dataset)
        train_set=dataset_module(
            img_paths=df.loc[df['fold']!=i,'path'].values,
            labels=df.loc[df['fold']!=i,'label'].values
        )
        val_set=dataset_module(
            img_paths=df.loc[df['fold']==i,'path'].values,
            labels=df.loc[df['fold']==i,'label'].values
        )

        num_classes=18

        transform_module = getattr(import_module("dataset"), args.augmentation)  # default: BaseAugmentation
        transform = transform_module(
            resize=args.resize,
            mean=train_set.mean,
            std=train_set.std,
        )
        test_transform_modele=getattr(import_module('dataset'),'CustomAugmentation')
        test_transform = transform_module(
            resize=args.resize,
            mean=train_set.mean,
            std=train_set.std,
        )
        train_set.set_transform(transform)
        val_set.set_transform(test_transform)

        weights=make_weights(train_set.labels,18)

        #sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))


        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            num_workers=multiprocessing.cpu_count() // 2,
            shuffle=False,
            pin_memory=use_cuda,
            drop_last=False,
            #sampler=sampler
        )

        val_loader = DataLoader(
            val_set,
            batch_size=args.valid_batch_size,
            num_workers=multiprocessing.cpu_count() // 2,
            shuffle=False,
            pin_memory=use_cuda,
            drop_last=False,
        )

        model_module = getattr(import_module("model"), args.model)  # default: BaseModel
        model = model_module(
            num_classes=num_classes
        ).to(device)
        model = torch.nn.DataParallel(model)

        criterion=F1Loss()
        opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
        optimizer = opt_module(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=5e-4
        )
        scheduler = StepLRScheduler(
            optimizer,
            decay_t=args.lr_decay_step,
            decay_rate=0.5,
            warmup_lr_init=2e-08,
            warmup_t=5,
            t_in_epochs=False,
        )
        
        patience=0
        best_val_acc = 0
        best_val_loss = np.inf
        for epoch in range(args.epochs):
            # train loop
            model.train()
            loss_value = 0
            matches = 0
            for idx, train_batch in enumerate(train_loader):
                inputs, labels = train_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outs=model(inputs)
                loss=criterion(outs,labels)

                loss.backward()
                optimizer.step()

                preds = torch.argmax(outs, dim=-1)
                loss_value += loss.item()
                matches += (preds == labels).sum().item()

                current_lr = get_lr(optimizer)
                if (idx + 1) % args.log_interval == 0:
                    train_loss = loss_value / args.log_interval
                    train_acc = matches / args.batch_size / args.log_interval
                    current_lr = get_lr(optimizer)
                    print(
                        f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                        f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                    )

                    loss_value = 0
                    matches = 0


            scheduler.step_update(epoch + 1)
            mask_total=[0,0,0]
            mask_correct=[0,0,0]

            gender_total=[0,0,0]
            gender_correct=[0,0,0]

            age_total=[0,0,0]
            age_correct=[0,0,0]

            target_list=[]
            pred_list=[]

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
                        
                        val_f1=f1_score(np.array(target_list),np.array(pred_list),average='macro')

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

                    print(f'age<30:{age_correct[0]/age_total[0]:4.2%}')
                    print(f'30<=age<60:{age_correct[1]/age_total[1]:4.2%}')
                    print(f'age>=60:{age_correct[2]/age_total[2]:4.2%}')

                    print(f'mask_wear:{mask_correct[0]/mask_correct[0]:4.2%}')
                    print(f'mask_incorrect{mask_correct[1]/mask_correct[1]:4.2%}')
                    print(f'mask_wrong{mask_correct[2]/mask_correct[2]:4.2%}')

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
                        patience+=1

                    torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
                    print(
                        f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                        f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2} ||"
                        f"f1_score: {val_f1:4.2%}"
                    )              



                    print()
            if patience==args.patience:
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')

    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type=str, default='fold_mask', help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--augmentation', type=str, default='CustomAugmentation', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument("--resize", nargs="+", type=list, default=[240, 240], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 64)')

    
    # Validation
    parser.add_argument('--validation_interval',type=int,default=10, help="Validation interval in training process (default: 10)")
    parser.add_argument('--valid_batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    

    parser.add_argument('--model', type=str, default='EfficientNet', help='model type (default: BaseModel)')
    
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: SGD)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-3)')

    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')
    parser.add_argument('--patience', type=int ,default=5,help='early stopping')
    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)
