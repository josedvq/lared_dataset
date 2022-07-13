from curses import window
import os
from pathlib import Path

import numpy as np
import cv2
from tqdm import tqdm

from ..preprocess.pose.utils import enlarge_rectangle, get_track_rectangle, tlwh_to_tlbr
from lared_dataset.constants import raw_videos_path

def make_examples(tracks, window_len=90, cam=0):
    examples = list()
    example_id = 0
    for _, track in enumerate(tracks):
        for i in range(0, len(track['poses']) // window_len):
            examples.append({
                'id': example_id,
                'pid': track['pid'],
                'cam': cam,
                'ini': track['ini'] + i * window_len,
                'poses': track['poses'][i * window_len: (i + 1) * window_len,:],
                'track_id': track['id'], 
                'track_ini': i * window_len, 
                'track_fin': (i + 1) * window_len
            })
            example_id += 1
    return examples

def reset_examples_ids(examples):
    for i, ex in enumerate(examples):
        ex['id'] = i

def filter_examples_by_movement_threshold(examples, ts=20):
    new_examples = list()

    for ex in examples:
        track = ex['poses']
        std_x = np.std(track[:,3])
        std_y = np.std(track[:,4])
        
        if std_x > ts or std_y > ts:
            continue

        new_examples.append(ex)

def write_example_video(ex, out_folder, cap):
    track = ex['poses'] 
    rect = get_track_rectangle(track)

    cap.set(cv2.CAP_PROP_POS_FRAMES, ex['ini']+1) # one-based
    vout_path = os.path.join(out_folder, '{:05d}.mp4'.format(ex['id']))
    
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
