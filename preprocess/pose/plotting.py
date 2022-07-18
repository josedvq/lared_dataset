import os
import cv2

from tqdm import tqdm
from matplotlib import pyplot as plt

from ...constants import raw_videos_path
from ...tools.pose_renderer import render_poses
from.pose_parameters import POSE_UPPER_BODY_PAIRS, POSE_UPPER_BODY_COLORS

class PosePlotter:

    def __init__(self):
        self.caps = {
            cam: cv2.VideoCapture(os.path.join(raw_videos_path, f'cam{cam}.mp4')) for cam in [1,2,3,4]
        }

    def plot_first_frame(self, example):
        # render the whole video
        
        ini_frame = example['ini']
        pose = example['poses'][0,:]


        cap = self.caps[example['cam']]
        cap.set(cv2.CAP_PROP_POS_FRAMES, ini_frame+1)
        ret, frame = cap.read()

        render_poses([pose], frame, pairs=POSE_UPPER_BODY_PAIRS, colors=POSE_UPPER_BODY_COLORS)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots()
        ax.imshow(rgb_frame)

    def plot_example_video(self, example, out_file):
        
        ini_frame = example['ini']
        end_frame = example['end']

        vout = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*'avc1'), 30, (1920,1080))

        cap = self.caps[example['cam']]
        cap.set(cv2.CAP_PROP_POS_FRAMES, ini_frame+1)

        for i in tqdm(range(ini_frame, end_frame)):
            # read a line with poses for every frame
            ret, frame = cap.read()
            pose = example['poses'][i,:]
            
            render_poses([pose], frame, pairs=POSE_UPPER_BODY_PAIRS, colors=POSE_UPPER_BODY_COLORS)

            vout.write(frame)
            
        vout.release()