import os
import pickle
from math import floor
from typing import Any, Callable, Optional
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.interpolate import interp1d
from pytorchvideo.data.video import VideoPathHandler

from .utils import (
    get_video_hash,
    video_seconds_to_accel_sample
)

class VideoExtractor():
    _MAX_CONSECUTIVE_FAILURES = 10

    def __init__(
        self,
        videos_path: str,
        transform: Optional[Callable[[dict], Any]] = None,
        sr = 30, # fps
        n_jobs = 1
    ) -> None:
        self.videos_path = videos_path
        self.transform = transform

        # If a RandomSampler is used we need to pass in a custom random generator that
        # ensures all PyTorch multiprocess workers have the same random seed.
        self._video_random_generator = None
        self.video_path_handler = VideoPathHandler()

        self.sr = sr
        self.n_jobs = n_jobs

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

        frames = video['video'].to(torch.uint8)
            
        if self.transform is not None:
            frames = self.transform(frames)

        return frames

    def __call__(self, key, start, end) -> dict:
        return self._get_clip(key, start, end)

    def extract_multiple(self, keys):
        return np.stack(Parallel(n_jobs=self.n_jobs)(delayed(self.__call__)(*k) for k in keys))
        return np.stack([self(*k) for k in keys])


class AccelExtractor():
    def __init__(self, accel_path, strict=False):
        self.accel = pickle.load(open(accel_path, 'rb'))
        self.strict = strict

        self.num_channels = 3
        self.fs = 20

    def __call__(self, pid, start, end):
        if self.strict and pid not in self.accel:
            raise ValueError(f'pid {pid} not in self.accel')
        
        if pid not in self.accel:
            return np.zeros((self.num_channels, round(self.fs * (end-start))), dtype=np.float32)

        accel_ini = video_seconds_to_accel_sample(start)
        accel_fin = video_seconds_to_accel_sample(end)

        my_subj_accel = self.accel[pid]
        ini_idx = np.argmax(my_subj_accel[:,0] > accel_ini)
        fin_idx = ini_idx + round(self.fs * (end-start))

        # if ini_idx == 0:
        #     print('out of bounds. pid={:d}, accel_ini={:.2f}'.format(ex['person'], accel_ini))

        accel = my_subj_accel[ini_idx: fin_idx, 1:]

        if len(accel) < round(self.fs * (end-start)):
            accel = np.pad(accel, 
                ((0, round(self.fs * (end-start))-len(accel)), (0, 0)),
                mode='constant',
                constant_values= 0)

        return accel.transpose().astype(np.float32)

    def extract_multiple(self, keys):
        return np.stack([self(*k) for k in keys])