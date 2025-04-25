import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset,DataLoader

from torchvision import transforms # nice: https://pytorch.org/vision/main/auto_examples/transforms/plot_transforms_illustrations.html#sphx-glr-auto-examples-transforms-plot-transforms-illustrations-py
#from torchvision.io import read_image

class CustomDataset(Dataset):
    def __init__(self, img_dir='training-images', depth_dir='training-depths', root_dir='competition-data', transform=transforms.ToTensor(), target_transform=transforms.ToTensor()):
        super().__init__()
        super().__init__()
        self.images = []
        self.labels = []
        for f in os.listdir(os.path.join(root_dir, img_dir)):
            self.images.append(os.path.join(root_dir, img_dir, f))
            self.labels.append(os.path.join(root_dir, depth_dir, f))
        
        self.transform = transform
        self.target_transform = target_transform


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        label = Image.open(self.labels[index])
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label




if __name__=="__main__":
    ds = CustomDataset(img_dir='training-images', depth_dir='training-depths', root_dir='competition-data')#, transform=transforms.ToTensor(), target_transform=transforms.ToTensor())
    dl = DataLoader(ds, batch_size=10)
    for x,y in dl:
        print(x.shape, y.shape)
        break
    
