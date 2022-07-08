import os
import random

import torch
import torchvision
import numpy as np
import pandas as pd

from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, DALIGenericIterator, LastBatchPolicy

@pipeline_def
def video_pipe(filenames):
    videos, labels, start_frame_num, timestamps = fn.readers.video(
        device="gpu", 
        filenames=filenames, 
        labels=[],
        sequence_length=1,
        shard_id=0, 
        num_shards=1, 
        random_shuffle=True, 
        initial_fill=32,
        enable_frame_num=True,
        enable_timestamps=True,
        seed=22)
    return videos, labels, start_frame_num, timestamps

class MnmDataloader(DALIGenericIterator):
    def __init__(self, videos_path, bounding_boxes_path, cameras_path, batch_size=2, spatial_res=244, *kargs, **kvargs):

        video_files=[videos_path + '/' + f for f in os.listdir(videos_path)]

        pipe = video_pipe(
            batch_size=batch_size,
            num_threads=2,
            device_id=0,
            filenames=video_files,
            seed=123456)
            
        super().__init__(pipe, ['videos', 'labels', 'start_frame_num', 'timestamps'], *kargs, **kvargs)
        self.batch_size = batch_size
        self.spatial_res = spatial_res

        # load bbs
        self.bbs = pd.read_csv(bounding_boxes_path, header=None, dtype=int)*2
        self.cam = pd.read_csv(cameras_path, header=None, dtype=int).to_numpy()
        self.num_subjects = len(self.bbs.columns) // 7

    def _video_frame_to_annot_frame(self, video_frame_number):
        return round(video_frame_number * 20 / 29.97)

    def _get_bbs_for_video_frame(self, video_frame_number, cam_number, num_bbs=1):
        fn = self._video_frame_to_annot_frame(video_frame_number)
        subjects = np.where(self.cam[fn, :] == cam_number)[0].tolist()
        subjects = random.sample(subjects, num_bbs)

        print(f'fn= {fn}')

        bbs = []
        for s in subjects:
            bb = self.bbs.iloc[fn, s*4: s*4+4].to_list()
            bbs.append([bb[1], bb[0], bb[3], bb[2]])

        return bbs

    def __next__(self):
        out = super().__next__()
        # DDP is used so only one pipeline per process
        # also we need to transform dict returned by DALIClassificationIterator to iterable
        # and squeeze the lables
        # return out

        start_frames = out[0]['start_frame_num'].cpu()
        labels = out[0]['labels'].cpu()
        videos = out[0]['videos'].cpu().permute(0,1,4,2,3)

        out_videos = torch.empty((self.batch_size, 3, self.spatial_res, self.spatial_res), dtype=int)
        for i in range(self.batch_size):
            bb = self._get_bbs_for_video_frame(start_frames[i].item(), labels[i].item())[0]
            print(bb)
            out_videos[i] = torchvision.transforms.functional.resized_crop(videos[i].squeeze(), *bb, 
                size=[self.spatial_res, self.spatial_res])
        
        return {'videos': out_videos}