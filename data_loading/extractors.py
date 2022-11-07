import os
import pickle
from math import floor
from typing import Any, Callable, Optional
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from pytorchvideo.data.video import VideoPathHandler

from utils import (
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


class AccelExtractor():
    def __init__(self, accel_path):
        self.accel = pickle.load(open(accel_path, 'rb'))

    def __call__(self, pid, start_time, end_time):
        assert pid in self.accel

        accel_ini = video_seconds_to_accel_sample(start_time)
        accel_fin = video_seconds_to_accel_sample(end_time)

        my_subj_accel = self.accel[pid]
        ini_idx = np.argmax(my_subj_accel[:,0] > accel_ini)
        fin_idx = np.argmax(my_subj_accel[:,0] > accel_fin) + 1

        if ini_idx == 0:
            print('out of bounds. pid={:d}, accel_ini={:.2f}'.format(ex['person'], accel_ini))

        accel = my_subj_accel[ini_idx: fin_idx, 1:]
        return accel
