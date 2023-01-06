import os
import pathlib

import torch
import torch.nn as nn
from slowfast.config.defaults import get_cfg
from slowfast.models.ptv_model_builder import PTVResNet, PTVSlowFast

from lared_dataset.constants import models_path

def _get_resnet_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(os.path.join(
        pathlib.Path(__file__).parent.resolve(), 'configs', 
        'SLOW_8x8_R50.yaml'))
    return cfg

def _make_pretrained_slow(head):
    cfg = _get_resnet_cfg()
    model = PTVResNet(cfg, head=head)

    chckpt = torch.load(os.path.join(models_path, 'SLOW_8x8_R50.pyth'))

    res = model.load_state_dict(chckpt['model_state'], strict=False)
    print(f'loaded pre-trained model')
    print(f'missing keys {str(res.missing_keys)}')
    print(f'unexpected keys {str(res.unexpected_keys)}')

    for param in model.parameters():
        param.requires_grad = False

    return model

def make_slow_pretrained_body(pool=True):
    if pool:
        head = nn.AvgPool3d(kernel_size=(8, 7, 7), stride=(1, 1, 1), padding=(0, 0, 0))
    else:
        head = nn.Sequential()
    return _make_pretrained_slow(head=head)

class VideoSegmentationHead(nn.Module):
    def __init__(self, output_len):
        super(VideoSegmentationHead, self).__init__()

        self.head = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 8, 8), stride=(1, 1, 1), padding=(0, 0, 0)),
            nn.Conv3d(2048, 1, (3,1,1), stride=1, padding=(1,0,0))
        )

        self.upsample = nn.Sequential(
            nn.Flatten(start_dim=2),
            nn.Upsample(size=(output_len), mode='linear')
        )

    def forward(self, x):
        x = self.head(x)
        assert x.shape[-1] == 1
        x = self.upsample(x)
        return x.squeeze()