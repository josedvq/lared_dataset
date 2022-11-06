import os
import pickle
from math import floor
from typing import Any, Callable, Optional
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from pytorchvideo.data.video import VideoPathHandler

from utils import get_video_hash

class VideoExtractor():
    _MAX_CONSECUTIVE_FAILURES = 10

    def __init__(
        self,
        videos_path: str,
        transform: Optional[Callable[[dict], Any]] = None,
        sr = 30, # fps
    ) -> None:
        self.videos_path = videos_path
        self.transform = transform

        # If a RandomSampler is used we need to pass in a custom random generator that
        # ensures all PyTorch multiprocess workers have the same random seed.
        self._video_random_generator = None
        self.video_path_handler = VideoPathHandler()

        self.sr = sr

    def _get_clip(self, pid, start, end):
        hash = get_video_hash(pid, start, end)

        video = self.video_path_handler.video_from_path(
            os.path.join(self.videos_path, f'{hash}.mp4'),
            decoder='pyav'
        )

        video = video.get_clip(0, float(video.duration))

        video_is_null = (video is None or video["video"] is None)
        
        if video_is_null:
            raise Exception(f"Failed to load clip {video.name}, for start={start}, end={end}")

        frames = video['video'].to(torch.uint32)
            
        if self.transform is not None:
            frames = self.transform(frames)

        return frames

    def __call__(self, key, start, end) -> dict:
        return self._get_clip(key, start, end)


class LabelExtractor():
    def __init__(self, annot_path, min_len=None, transform=None):
        self.annot_path = annot_path
        self.transform = transform
        self.min_len = min_len
        

        self.annot = {}
        for f in Path(self.annot_path).glob('[0-9]{1-2}'):
            self.annot[int(f.name)] = pd.read_csv(f, header=None, index_col=False)

    def __call__(self, pid, start_time, end_time):
        start = floor(start_time * 100) 
        end = floor(end_time * 100)
        subject_annot = self.annot[pid].iloc[start: end, :]

        if self.min_len is not None and example_annot.shape[0] < self.min_len:
            example_annot = np.pad(example_annot, 
                (0, self.min_len-example_annot.shape[0]),
                mode='constant',
                constant_values= 0)

        if self.transform:
            example_annot = self.transform(example_annot)

        return example_annot


class SegMaskExtractor():
    def __init__(self, annot_path, transform=None, min_len=None, max_len=None, sr=30):
        self.annot_path = annot_path
        self.transform = transform
        if min_len is not None and max_len is not None:
            assert max_len >= min_len
        self.min_samples = round(sr*min_len) if min_len is not None else None
        self.max_samples = round(sr*max_len) if max_len is not None else None
        
        self.sr = sr
        
        self.annot = pickle.load(open(annot_path, 'rb'))

    def _subsample(self, annot, window: Tuple[int, int]):
        start = max(0, round(window[0] * self.sr))
        end  = min(len(annot), round(window[1] * self.sr))

        if len(annot) < end - start:
            raise Exception(f'Not implemented. annot shape is {str(annot.shape)}')

        annot = annot[start: end]
        return annot

    def extract_multiple(self, keys):
        return np.stack([self(*k) for k in keys])

    def __call__(self, key, start=None, end=None):
        assert (start is None and end is None) or (start is not None and end is not None)

        example_annot = self.annot[key].astype(np.float32)

        if start is not None and end is not None:
            example_annot = self._subsample(example_annot, (start, end))

        if self.min_samples is not None and len(example_annot) < self.min_samples:
            example_annot = np.pad(example_annot, 
                ((0, self.min_samples - len(example_annot))),
                mode='constant',
                constant_values= 0)

        if self.max_samples is not None and len(example_annot) > self.max_samples:
            example_annot = example_annot[:self.max_samples]

        if self.transform:
            return self.transform(example_annot)

        return example_annot