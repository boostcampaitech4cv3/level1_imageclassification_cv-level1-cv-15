from importlib import import_module
from dataset import ImageDataset
from torch.utils.data import DataLoader
from .sampler import RandomIdentitySampler
import multiprocessing
import torch

def train_collate_fn(batch):

    imgs, pids = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids

def val_collate_fn(batch):
    imgs, pids = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids


def make_dataloader(data_dir,cfg):
    
    # -- dataset
    dataset_module = getattr(import_module("dataset"), cfg.dataset)  # default: MaskBaseDataset
    dataset = dataset_module(
        data_dir=data_dir,
    )

    num_classes = dataset.num_classes  # 18

    # -- augmentation
    transform_module = getattr(import_module("dataset"), cfg.augmentation)  # default: BaseAugmentation , CustomAugmentation
    transform = transform_module(
        resize=cfg.resize,
        cropsize = cfg.cropsize,
        mean=dataset.mean,
        std=dataset.std,
    )

    train_set = ImageDataset(dataset.train_image_with_ID, transform) 
    val_set = ImageDataset(dataset.val_image_with_ID, transform)

    if 'triplet' in cfg.sampler:
        train_loader = DataLoader( 
            train_set, 
            batch_size=cfg.batch_size,
            sampler=RandomIdentitySampler(dataset.train_image_with_ID, cfg.batch_size, cfg.num_instance),
            num_workers=multiprocessing.cpu_count() // 2, 
            collate_fn=train_collate_fn,
            pin_memory=torch.cuda.is_available(),
            drop_last = True
        ) 
    elif cfg.sampler == 'softmax':
        print('using softmax sampler')
        train_loader = DataLoader(
            train_set, 
            batch_size=cfg.batch_size, 
            num_workers=multiprocessing.cpu_count() // 2,
            shuffle=True, 
            collate_fn=train_collate_fn,
            pin_memory = torch.cuda.is_available(),
            drop_last = True
        )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.sampler))

    val_loader = DataLoader(
        val_set,
        batch_size=cfg.valid_batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    
    return train_loader, val_loader , num_classes