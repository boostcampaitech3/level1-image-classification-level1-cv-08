import argparse
import os
from importlib import import_module
import glob
from tqdm import tqdm
import numpy as np 
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import TestDataset, MaskBaseDataset, CustomDataset, CustomMaskBaseDataset
from model import *

from torchvision import transforms
from torchvision.transforms import *
from PIL import Image


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
    # 1. cuda settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    

    # 2. make dataset
    img_root = os.path.join(data_dir, 'images') # 'input/data/eval/images'
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)
    
    img_paths = [os.path.join(img_root,img_id) for img_id in info.ImageID]
    test_transform = transforms.Compose([
            ToPILImage(),
            CenterCrop((300, 256)),
            Resize(args.resize, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    dataset = CustomDataset(img_paths, training=False)
    dataset.set_transform(test_transform)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )
    
    model_pred_lst = []
    fold = 0
    pthfiles = glob.glob(f'{args.model_dir}/{args.name}/*.pth')
    pthfiles = sorted(pthfiles)
    print(pthfiles)
    for best_model in pthfiles:
        model_module = getattr(import_module("model"), args.model)
        model = model_module(18)
        model.load_state_dict(torch.load(best_model))
        model.to(device)
        model.eval()
        
        small_pred_lst = []
        test_bar = tqdm(loader, total=loader.__len__(), unit='batch') 
        for item in test_bar:
            imgs = item['image'].float().to(device)
            pred = model(imgs)
            pred = pred.cpu().detach().numpy()
            small_pred_lst.extend(pred)
        model_pred_lst.append(np.array(small_pred_lst)[...,np.newaxis])
        
    print("Calculating inference results..")
    info['ans'] = np.argmax(np.mean(np.concatenate(model_pred_lst, axis=2), axis=2), axis=1)
    info.to_csv(os.path.join(output_dir, f'submission.csv'), index=False)
    print(f'Inference Done!')

    del model, imgs
    torch.cuda.empty_cache()
    
class InConfig:
    batch_size = 200
    resize = (96,128)
    model = "MyEfficientNet"
    name = "exp"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=InConfig.batch_size, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', type=tuple, default=InConfig.resize, help='resize size for image when you trained (default: (96, 128))')
    parser.add_argument('--model', type=str, default=InConfig.model, help='model type (default: BaseModel)')
    parser.add_argument('--name', type=str, default=InConfig.name, help='folder name')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dir, output_dir, args)
