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

from mixup import mixup,mixed_criterion
from timm.scheduler.step_lr import StepLRScheduler
from sklearn.metrics import f1_score

# from torch.utils.tensorboard import SummaryWriter 

from dataset import MaskBaseDataset   # dataset class import
from loss import create_criterion,F1Loss
from model import BaseModel, ResNet34, ResNet152  # model.py에서 model class import

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


def train(data_dir, model_dir, args):
    seed_everything(args.seed)

    # model 이란 폴더 안에서 하위폴더의 path index를 매 experiment마다 늘려줌
    save_dir = increment_path(os.path.join(model_dir, args.name))

    # 위의 save_dir 폴더 만들기 
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    # 현재 arguments값을 config.json파일로 dump하기(나중에 hyperparameter값을 알기 위해)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)
    
    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    dataset_module = getattr(import_module("dataset"), args.dataset)  # default: MaskBaseDataset
    dataset = dataset_module(
        data_dir=data_dir,
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

    # -- data_loader
    train_set, val_set = dataset.split_dataset() # dataset

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True,
    )

    # -- model
    model_module = getattr(import_module("model"), args.model)  # default: BaseModel
    model = model_module(
        num_classes=num_classes
    ).to(device)
    model = torch.nn.DataParallel(model)

    # -- loss & metric
    #criterion = create_criterion(args.criterion)  # default: cross_entropy
    #criterion=FocalLoss(gamma=2)
    criterion=FocalLoss(gamma=2)
    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=5e-4
    )
    #optimizer=AdamP(model.parameters(),lr=args.lr)

    #scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)
    scheduler = StepLRScheduler(
            optimizer,
            decay_t=args.lr_decay_step,
            decay_rate=0.5,
            warmup_lr_init=2e-08,
            warmup_t=5,
            t_in_epochs=False,
        )
    # -- Tensorboard logging
    # logger = SummaryWriter(log_dir=save_dir)



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
            
            #v=random.randint(1,2)
            '''
            if v==1:
                lam=np.random.beta(0.2,0.2)
                inputs,y_a,y_b=mixup(inputs,labels,lam)
                outs = model(inputs)          
            #loss = criterion(outs, labels)
                loss=mixed_criterion(criterion,outs,y_a,y_b,lam)
            else:
                outs=model(inputs)
                loss=criterion(outs,labels)
            '''
            outs=model(inputs)
            loss=criterion(outs,labels)

            loss.backward()
            optimizer.step()

            preds = torch.argmax(outs, dim=-1)

            loss_value += loss.item()
            matches += (preds == labels).sum().item()
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
        #if not (epoch + 1) % args.validation_interval : # Validation 하는 주기는 알아서 바꿔서 해도 될듯!
        
        # val loop

        #if not (epoch + 1) % args.validation_interval : # Validation 하는 주기는 알아서 바꿔서 해도 될듯!
        # val loop
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



                    if figure is None:
                        inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                        inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                        figure = grid_image(
                            inputs_np, labels, preds, n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
                        )
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
                torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
                print(
                    f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                    f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2} ||"
                    f"f1_score: {val_f1:4.2%}"
                )            


                # Tensorboard
                # logger.add_scalar("Val/loss", val_loss, epoch)
                # logger.add_scalar("Val/accuracy", val_acc, epoch)
                # logger.add_figure("results", figure, epoch)
                print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')

    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type=str, default='MaskSplitByProfileDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--augmentation', type=str, default='CustomAugmentation', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument("--resize", nargs="+", type=list, default=[240, 240], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')

    
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

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)
