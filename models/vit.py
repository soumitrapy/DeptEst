from torch.nn import TransformerEncoderLayer
import torch.nn as nn
import torch
class VIT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encs = nn.Sequential(*[TransformerEncoderLayer(d_model=config['model']['d_model'], nhead=config['model']['nhead'], batch_first=True) for _ in range(config['model']['n_encoder_layers'])])
    
    def forward(self, x):
        x = x.squeeze(1)
        enc = self.encs(x)
        return enc.unsqueeze(1)

if __name__=="__main__":
    x = torch.randn(10, 1, 256, 256)
    config = {'model': {'n_encoder_layers': 1,
                        'd_model': 256,
                        'nhead': 8
                        }
                }
    m = VIT(config)
    print(m(x).shape)
