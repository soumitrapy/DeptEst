import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class DoubleConvolution(nn.Module):
    def __init__(self,in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act2 = nn.ReLU()
    
    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        return x
class CropandConcat(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x: torch.Tensor, contracting_x: torch.Tensor):
        contracting_x = transforms.CenterCrop(size=(x.shape[-2], x.shape[-1]))(contracting_x)
        x = torch.cat([x, contracting_x], dim = 1)
        return x

class UNet(nn.Module):
    def __init__(self,config):
        super().__init__()
        cfg = config['model']
        self.filters = [1, 64, 128, 256, 512, 1024]
        if cfg.get('filters', False):
            self.filters = cfg['filters']
        self.dconvs = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i in range(len(self.filters)-1):
            self.dconvs.append(DoubleConvolution(self.filters[i], self.filters[i+1]))
            if i==len(self.filters)-2:
                self.downs.append(None)
            else:
                self.downs.append(nn.MaxPool2d(kernel_size=2))
        
        self.pups = nn.ModuleList()
        for i in range(1, len(self.filters)):
            self.pups.append(nn.ConvTranspose2d(self.filters[i], self.filters[0], kernel_size=2**(i-1), stride=2**(i-1)))
        
        self.ups = nn.ModuleList()
        self.crcs = nn.ModuleList()
        self.dconvs2 = nn.ModuleList()
        for i in range(len(self.filters)-1, 1, -1):
            self.ups.append(nn.ConvTranspose2d(self.filters[i], self.filters[i-1], kernel_size = 2, stride=2))
            self.crcs.append(CropandConcat())
            self.dconvs2.append(DoubleConvolution(self.filters[i-1]*2, self.filters[i-1]))
        self.last_conv = nn.Conv2d(in_channels=self.filters[1], out_channels=self.filters[0], kernel_size=1)

        

    def forward(self, x):
        outs = []
        partials = []
        for dconv, down in zip(self.dconvs, self.downs):
            x = dconv(x)
            if down is not None:
                outs.append(x)
                partials.append(x)
                x = down(x)

        partials.append(x)
        for i, (up, crc, dconv) in enumerate(zip(self.ups, self.crcs, self.dconvs2)):
            x = up(x)
            x = crc(x, outs[-i-1])
            x = dconv(x)
            partials.append(x)
        x = self.last_conv(x)
        x = torch.sigmoid(x)

        pouts = []
        for i in range(1,len(self.filters)):
            o1, o2 = self.pups[i-1](partials[i-1]), self.pups[i-1](partials[-i])
            pouts.extend([o1,o2])
  
        return pouts

if __name__=="__main__":
    config = {'model':{'filters':[1, 4, 8, 16, 32, 64]}}
    m = UNet(config)
    x = torch.randn(5,1, 256, 256)
    #xx = torch.randn(10, 4, 342, 268)
    for t in m(x):
        print(t.shape)
