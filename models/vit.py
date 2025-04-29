from torch.nn import TransformerEncoderLayer
import torch.nn as nn
import torch
import torch.nn.functional as F

class VIT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encs = nn.Sequential(*[TransformerEncoderLayer(d_model=config['model']['d_model'], nhead=config['model']['nhead'], batch_first=True) for _ in range(config['model']['n_encoder_layers'])])
    
    def forward(self, x):
        x = self.im2v(x)
        enc = self.encs(x)
        return self.v2im(enc)

    def im2v(self, x): # (B, 1, 256, 256)
        t = F.unfold(x, kernel_size=self.config['model']['patch_size'], stride=self.config['model']['patch_size']) # shape: (B, C * 16 * 16, N)
        return t.transpose(-2, -1)
    def v2im(self, p):
        p = p.transpose(-2,-1)
        return F.fold(p, output_size = (self.config['dataset']['img_size'], self.config['dataset']['img_size']), kernel_size=self.config['model']['patch_size'], stride=self.config['model']['patch_size'])


class VITInput:
    def __init__(self, image_size=256, patch_size = 16):
        self.image_size = image_size
        self.patch_size = patch_size
    def im2v(self, x): # (B, 1, 256, 256)
        t = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size) # shape: (B, C * 16 * 16, N)
        return t.transpose(-2, -1)
    def v2im(self, p):
        p = p.transpose(-2,-1)
        return F.fold(p, output_size = (self.image_size, self.image_size), kernel_size=self.patch_size, stride=self.patch_size)
    

# # Assume you already have these:
# images = torch.randn(10, 1, 256, 256)
# patches = F.unfold(images, kernel_size=16, stride=16)  # shape: (10, 256, 256)

# # Now reconstruct:
# reconstructed = F.fold(
#     patches,                     # shape: (B, C*16*16, N)
#     output_size=(256, 256),      # original image size
#     kernel_size=16,
#     stride=16
# )

# print(reconstructed.shape)  # (10, 1, 256, 256)
# torch.allclose(images, reconstructed)  # Should be True
if __name__=="__main__":
    x = torch.randn(10, 1, 256, 256)
    config = {'model': {'n_encoder_layers': 1,
                        'd_model': 256,
                        'nhead': 8
                        }
                }
    m = VIT(config)
    print(m(x).shape)
