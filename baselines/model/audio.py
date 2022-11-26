import os
from collections import OrderedDict

import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=1):
        super().__init__()
        
        # Conv Layer 1
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=(3, 3), stride=stride, padding=padding, bias=True
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Conv Layer 2
        self.conv2 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=(3, 3), stride=1, padding=padding, bias=True
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
    
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels, out_channels=out_channels,
                    kernel_size=(1, 1), stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = nn.ReLU()(out)
        return out
    

class ResNetBody(nn.Module):
    def __init__(self,
        filter_sizes=[64,32,16,16]):
        super().__init__()

        # Initial input conv
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=64, kernel_size=(3, 3),
            stride=1, padding=1, bias=False
        )

        self.bn1 = nn.BatchNorm2d(64)
        
        self.filter_sizes = filter_sizes

        self.block1 = self._create_block(64, filter_sizes[0], stride=1)
        self.block2 = self._create_block(filter_sizes[0], filter_sizes[1], stride=2)
        self.block3 = self._create_block(filter_sizes[1], filter_sizes[2], stride=2)
        self.block4 = self._create_block(filter_sizes[2], filter_sizes[3], stride=2)
    
    # A block is just two residual blocks for ResNet18
    def _create_block(self, in_channels, out_channels, stride):
        return nn.Sequential(
            ResidualBlock(in_channels, out_channels, stride),
            ResidualBlock(out_channels, out_channels, 1)
        )

    def forward(self, x):
    # Output of one layer becomes input to the next
        x = nn.ReLU()(self.bn1(self.conv1(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x

def get_pretrained_body():
    model = ResNetBody()
        
    dir_path = os.path.dirname(os.path.realpath(__file__))
    chkpt = torch.load(os.path.join(dir_path, 
        './pretrained_audioset/audioset-epoch=09-val_loss=0.46.ckpt'))

    sd = OrderedDict()
    for el in chkpt['state_dict']:
        sd[el[6:]] = chkpt['state_dict'][el]

    res = model.load_state_dict(sd, strict=False)
    print(f'loaded pre-trained model')
    print(f'missing keys {str(res.missing_keys)}')
    print(f'unexpected keys {str(res.unexpected_keys)}')

    for param in model.parameters():
        param.requires_grad = False

    return model

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=1):
        super().__init__()
        
        # Conv Layer 1
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=(3, 3), stride=stride, padding=padding, bias=True
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Conv Layer 2
        self.conv2 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=(3, 3), stride=1, padding=padding, bias=True
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
    
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels, out_channels=out_channels,
                    kernel_size=(1, 1), stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = nn.ReLU()(out)
        return out

class AudioSegmentationHead(nn.Module):
    def __init__(self, num_channels, output_len, num_classes=1):
        super().__init__()
        self.head = nn.Sequential(
            ResidualBlock(num_channels, 8, stride=(1,2)),
            ResidualBlock(8, num_classes, stride=(1,2)),
            nn.AvgPool2d((1,4), stride=1),
        )

        self.upsample = nn.Sequential(
            nn.Flatten(start_dim=2),
            nn.Upsample(size=(output_len), mode='linear')
        )

    def forward(self, x):
        x = self.head(x)
        # at this point the output should have size=1 along the mel dimension
        assert x.shape[-1] == 1
        x = self.upsample(x)
        return x.squeeze()