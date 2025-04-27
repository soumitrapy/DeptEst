import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset,DataLoader

from torchvision import transforms # nice: https://pytorch.org/vision/main/auto_examples/transforms/plot_transforms_illustrations.html#sphx-glr-auto-examples-transforms-plot-transforms-illustrations-py
#from torchvision.io import read_image

class CustomDataset(Dataset):
    def __init__(self, img_dir='training-images', depth_dir=None, root='competition-data', transform=transforms.ToTensor(), target_transform=transforms.ToTensor()):
        super().__init__()
        super().__init__()
        self.images = []
        self.labels = []
        for f in os.listdir(os.path.join(root, img_dir)):
            self.images.append(os.path.join(root, img_dir, f))
            if depth_dir is not None:
                self.labels.append(os.path.join(root, depth_dir, f))
        
        self.transform = transform
        self.target_transform = target_transform


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        if self.transform:
            image = self.transform(image)
        if len(self.labels)==0:
            return image, self.images[index] # In the case of test set it will return image tensor and the images names as tuple
        
        else:
            label = Image.open(self.labels[index])
            if self.target_transform:
                label = self.target_transform(label)
            return image, label
        

def get_datasets_and_dataloaders(config):
    cfg = config['dataset']
    ds = {'train': CustomDataset(img_dir=cfg['training-images'], depth_dir=cfg['training-depths'], root=cfg['root'], transform=transforms.ToTensor(), target_transform=transforms.ToTensor()),
          'val': CustomDataset(img_dir=cfg['validation-images'], depth_dir=cfg['validation-depths'], root=cfg['root'], transform=transforms.ToTensor(), target_transform=transforms.ToTensor()),
          'test': CustomDataset(img_dir=cfg['testing-images'], root=cfg['root'], transform=transforms.ToTensor())
          }
    dl = {key: DataLoader(data, batch_size=cfg['batch_size'], shuffle=True) for key, data in ds.items()}
    return ds, dl

if __name__=="__main__":
    ds = CustomDataset(img_dir='testing-images', root='competition-data')#, transform=transforms.ToTensor(), target_transform=transforms.ToTensor())
    dl = DataLoader(ds, batch_size=5)
    for x, n in dl:
        print(x.shape)
        break
    
