import torch

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample
)

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    CenterCrop,
    RandomHorizontalFlip
)

class NoneTransform(object):
    """ Does nothing to the image, to be used instead of None
    
    Args:
        image in, image out, nothing is done
    """
    def __call__(self, image):       
        return image

# https://pytorch.org/hub/facebookresearch_pytorchvideo_slowfast/
class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors. 
    """
    def __init__(self, slowfast_alpha):
        super().__init__()
        self.slowfast_alpha = slowfast_alpha
        
    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // self.slowfast_alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list

def get_kinetics_train_transform(subsample_frames=8, spatial_res=244, pack=False):
    return Compose([
        ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                UniformTemporalSubsample(subsample_frames),
                Lambda(lambda x: x / 255.0),
                Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                RandomShortSideScale(min_size=256, max_size=320),
                RandomCrop(spatial_res),
                RandomHorizontalFlip(p=0.5),
                PackPathway(4) if pack else NoneTransform()
                ]
            ),
            ),
    ])

def get_kinetics_val_transform(subsample_frames=8, spatial_res=244, pack=False):
    return Compose(
        [
        UniformTemporalSubsample(subsample_frames),
        Lambda(lambda x: x / 255.0),
        Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
        ShortSideScale(256),
        CenterCrop(spatial_res),
        PackPathway(4) if pack else NoneTransform()
        ]
    )