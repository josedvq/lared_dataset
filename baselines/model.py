from sklearn.metrics import f1_score
import torch
from pytorch_lightning.utilities.seed import isolate_rng

from .video.models import (
    make_slow_pretrained_body as get_video_feature_extractor)
from .accel.resnet import (
    ResNetBody as AccelModelBody,
    ResNetBodyNoChannelPool as AccelModelBodyNoPool, 
)

class FusionModel(torch.nn.Module):
    def __init__(self,
        modalities):
        """
        """
        super().__init__()

        feature_sizes = {
            'video': 8192,
            'accel': 128
        }

        if 'video' in modalities:
            self.video_feature_extractor = get_video_feature_extractor().cuda()

        if 'accel' in modalities:
            self.accel_feature_extractor = AccelModelBody(c_in=3).cuda()

        self.modalities = modalities

        num_features = sum([feature_sizes[m] for m in modalities])
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(num_features, 1)
        )

    def forward(self, batch:dict):
        """
        """
        features = []

        if 'accel' in batch:
            features.append(self.accel_feature_extractor(batch['accel']))

        if 'video' in batch:
            features.append(self.video_feature_extractor(batch['video']))
        
        x = self.linear(torch.hstack(features))

        return x
        
