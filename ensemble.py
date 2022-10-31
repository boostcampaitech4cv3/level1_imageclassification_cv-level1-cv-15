import argparse
import multiprocessing
import os
from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import TestDataset, MaskBaseDataset


def load_model(saved_model, num_classes, device):
    model_cls = getattr(import_module("model"), args.model)
    model = model_cls(
        num_classes=num_classes
    )

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    model_path = os.path.join(saved_model, 'best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = MaskBaseDataset.num_classes  # 18
    #model = load_model(model_dir, num_classes, device).to(device)
    model_list=[]
    for m in model_dir:
        model_list.append(load_model(m,num_classes,device).to(device))

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device)
            pred = model_list[0](images)

            for  i in range(1,5):
                pred+=model_list[i](images)
            pred = pred.argmax(dim=-1)
            preds.extend(pred.cpu().numpy())

    info['ans'] = preds
    save_path = os.path.join(output_dir, f'output.csv')
    info.to_csv(save_path, index=False)
    print(f"Inference Done! Inference result saved at {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', type=tuple, default=(240, 240), help='resize size for image when you trained (default: (96, 128))')
    parser.add_argument('--model', type=str, default='EfficientNet', help='model type (default: BaseModel)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir1', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model/exp3'))
    parser.add_argument('--model_dir2', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model/exp4'))
    parser.add_argument('--model_dir3', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model/exp5'))
    parser.add_argument('--model_dir4', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model/exp6'))
    parser.add_argument('--model_dir5', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model/exp7'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))


    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = [args.model_dir1,args.model_dir2,args.model_dir3,args.model_dir4,args.model_dir5]
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dir, output_dir, args)