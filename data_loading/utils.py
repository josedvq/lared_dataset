import os
import hashlib
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from scipy.interpolate import interp1d

from ..preprocess.pose.utils import enlarge_rectangle, get_track_rectangle, tlwh_to_tlbr
from lared_dataset.constants import (
    raw_videos_path,
    balloon_pop_1_video_frame,
    balloon_pop_1_accel_frame,
    balloon_pop_3_video_frame,
    balloon_pop_3_accel_frame
)

def get_video_hash(pid, ini_time, end_time):
    return hashlib.md5(f'{pid}_{ini_time:.2f}_{end_time:.2f}'.encode()).hexdigest()

def write_example_video(ex, out_folder, cap):
    track = ex['poses'] 
    rect = get_track_rectangle(track)

    cap.set(cv2.CAP_PROP_POS_FRAMES, ex['ini']+1) # one-based
    vout_path = os.path.join(out_folder, '{:s}.mp4'.format(ex['hash']))
    if os.path.exists(vout_path):
        return
    
    # enlarge the rectangle
    # w_to_add = round(24 + 15 * rect[1] / 1080)
    # h_to_add = round(24 + 15 * rect[1] / 1080)
    # x1 = max(0, rect[0] - w_to_add)
    # y1 = max(0, rect[1] - h_to_add)
    # x2 = min(1920, rect[0] + rect[2] + w_to_add)
    # y2 = min(1080, rect[1] + rect[3] + h_to_add)
    rect = enlarge_rectangle(
        rect, 
        perc=0.15, 
        cast_to_int=True)

    x1, y1, x2, y2 = tlwh_to_tlbr(rect)
    
    # new size
    w = 160
    h = round(160 * (y2-y1) / (x2-x1))
    vout = cv2.VideoWriter(vout_path, cv2.VideoWriter_fourcc(*'avc1'), 30, (w, h))
    
    # write the video
    for i in range(0, len(track)):
        # draw the poses
        ret, frame = cap.read()
        f = frame[y1:y2, x1:x2, :]
        
        vout.write(cv2.resize(f, (w, h)))
    vout.release()

def write_all_example_videos(examples, out_folder):
    caps = {
        cam: cv2.VideoCapture(os.path.join(raw_videos_path, f'cam{cam}.mp4')) for cam in [1,2,3,4]
    }

    for ex in tqdm(examples):
        write_example_video(ex, out_folder, caps[ex['cam']])

def reset_examples_ids(examples):
    for i, ex in enumerate(examples):
        ex['id'] = i

class Maker():

    def __init__(self, tracks_path=None, accel_path=None, vad_path=None):
        self.tracks = pickle.load(open(tracks_path, "rb"))

        self.accel = {}
        if accel_path is not None:
            self.load_accel(accel_path)

        self.vad = {}
        if vad_path is not None:
            self.load_vad(vad_path)

        self.examples = None

    def load_accel(self, accel_path):
        self.accel = pickle.load(open(accel_path, 'rb'))

    def load_vad(self, vad_path):
        self.vad = {}
        for i in range(1, 45):
            fpath = os.path.join(vad_path, f'{i}.vad')
            if os.path.exists(fpath) and os.path.isfile(fpath):
                self.vad[i] = pd.read_csv(fpath, header=None).to_numpy()

        if len(self.vad) == 0:
            print('load_vad called but nothing loaded.')

    def _get_vad(self, pid, ini_time, end_time, vad_fs=100):
        # note audio (vad) and video start at the same time
        if pid not in self.vad:
            return None

        ini = round(ini_time * vad_fs)
        width = round((end_time - ini_time) * vad_fs)
        end = ini + width  

        return self.vad[pid][ini:end].flatten()

    def _interp_vad(self, vad, in_fs, out_fs):
        t = np.arange(0, len(vad) / in_fs, 1/in_fs)
        f = interp1d(t, vad, kind='nearest')
        tnew = np.arange(0, len(vad) / in_fs, 1/out_fs)
        return f(tnew)

    def make_examples(self, window_len=90, cam=0):
        examples = list()
        example_id = 0
        for _, track in enumerate(self.tracks):
            for i in range(0, len(track['poses']) // window_len):

                if track['pid'] not in self.accel:
                    continue

                ini_time = (track['ini'] + i * window_len) / 29.97
                end_time = (track['ini'] + (i+1) * window_len) / 29.97

                hash = get_video_hash(track["pid"], ini_time, end_time)
                vad = self._get_vad(track['pid'], ini_time, end_time)
                interp_vad = self._interp_vad(vad, 100, 20)

                if vad is None:
                    continue

                poses = track['poses'][i * window_len: (i + 1) * window_len,:]
                rect = get_track_rectangle(poses)
                if rect[2] == 0 or rect[3] == 0:
                    continue
                
                examples.append({
                    'id': example_id,
                    'pid': track['pid'],
                    'cam': cam,
                    'ini': track['ini'] + i * window_len,
                    'hash': hash,
                    'ini_time': ini_time,
                    'end_time': end_time,
        
                    # track info
                    'track_id': track['id'], 
                    'track_ini': i * window_len, 
                    'track_fin': (i + 1) * window_len,

                    # data
                    'poses': poses,
                    'vad': vad,
                    'interp_vad': interp_vad
                })
                example_id += 1
        
        self.examples = examples
        return examples

    def filter_examples_by_movement_threshold(self, ts=20):
        new_examples = list()

        for ex in self.examples:
            track = ex['poses']
            std_x = np.std(track[:,3])
            std_y = np.std(track[:,4])
            
            if std_x > ts or std_y > ts:
                continue

            new_examples.append(ex)

    
video_seconds_to_accel_sample = interp1d(
    [
        balloon_pop_1_video_frame/29.97, 
        balloon_pop_3_video_frame/29.97
    ], [
        balloon_pop_1_accel_frame, 
        balloon_pop_3_accel_frame
    ], fill_value="extrapolate")