import json
import os
import logging
import argparse
import math
import ast

import numpy as np
from tqdm import tqdm
import pickle
import cv2
import joblib
from joblib import Parallel, delayed
from dynarray import DynamicArray
from sklearn.metrics.pairwise import euclidean_distances
from scipy.optimize import linear_sum_assignment

from ...tools import pose_renderer
from jose.helpers.tqdm import tqdm_joblib


class FramePoses:

    next_id = 0
    img_w = 1920
    img_h = 1080

    def __init__(self, keypoints, frame_num=None):
        self.frame_num = frame_num
        self.keypoints = keypoints
        self.track_ids = [None] * len(keypoints)
        self.track_next= [None] * len(keypoints)
        self.track_prev= [None] * len(keypoints)
        self.track_len = [0] * len(keypoints)

    @classmethod
    def from_openpose_json(cls, path):
        keypoints = list()
        with open(path) as json_file:
            data = json.load(json_file)
            for p in data['people']:
                keypoints.append(p['pose_keypoints_2d'])
        return cls(keypoints)

    @classmethod
    def from_openpose_text(cls, path):
        pose_file = open(path, 'r')
        all_frame_poses = list()

        line = pose_file.readline()
        while line:
            keypoints = ast.literal_eval(line)
            all_frame_poses.append(cls(keypoints))
            line = pose_file.readline()
        return all_frame_poses

    @staticmethod
    def from_openpose_text_pruducer(path):
        pose_file = open(path, 'r')
        i = 0
        lines = list()

        line = pose_file.readline()
        while line:
            lines.append(line)
            line = pose_file.readline()
            i += 1
            if i % 5000 == 0 or not line:
                # print('yielding, line={:d}'.format(i))
                yield lines
                lines = list()

    @staticmethod
    def parse_lines(lines):
        all_frame_poses = list()
        for line in lines:
            keypoints = np.array(ast.literal_eval(line), np.float32)
            all_frame_poses.append(keypoints)
        return all_frame_poses

    @classmethod
    def from_openpose_text_fast(cls, path):
        with tqdm_joblib(tqdm(desc="Loading poses")) as progress_bar:
            poses = Parallel(n_jobs=8, verbose=1)(delayed(FramePoses.parse_lines)(lines)
                           for lines in FramePoses.from_openpose_text_pruducer(path))
        return poses

    def __len__(self):
        return len(self.keypoints)
    
    def __str__(self):
        return str(self.keypoints)

    @staticmethod
    def get_all_ended_tracks(prev_frame_poses):
        if len(prev_frame_poses) == 0:
            return list()

        ended_tracks = list()
        # output ended tracks, with no next pose
        for i in range(0, len(prev_frame_poses[-1])):
            # if the track has an id and is ended
            if prev_frame_poses[-1].track_ids[i] is not None and prev_frame_poses[-1].track_next[i] is None:
                assert prev_frame_poses[-1].track_len[i] > 0

                track_len = prev_frame_poses[-1].track_len[i]
                track_id  = prev_frame_poses[-1].track_ids[i]
                
                # create the np array to hold the pose sequence
                ps = np.empty((track_len, 13*3), dtype=np.float32)

                # add the current pose
                ps[-1,:] = np.concatenate([prev_frame_poses[-1].keypoints[i][0*3:9*3], prev_frame_poses[-1].keypoints[i][15*3:19*3]])

                # follow the prev links to extract the rest of the pose sequence
                j = -1
                pt = i
                # while there is a link to a prev pose
                while prev_frame_poses[j].track_prev[pt] is not None:
                    next_pt = pt
                    pt = prev_frame_poses[j].track_prev[pt]
                    j -= 1

                    assert prev_frame_poses[j].track_ids[pt] == track_id
                    assert prev_frame_poses[j].track_len[pt] == track_len + j + 1
                    assert prev_frame_poses[j].track_next[pt] == next_pt

                    # add the pose
                    ps[j,:] = np.concatenate([prev_frame_poses[j].keypoints[pt][0*3:9*3], prev_frame_poses[j].keypoints[pt][15*3:19*3]])
                
                frame_num = prev_frame_poses[j].frame_num
                ended_tracks.append({'id': track_id, 'ini': frame_num, 'poses': ps})

        return ended_tracks

    def assign_tracks(self, prev_frame_poses, tracks):
        kp = 1
        for i in range(0, len(self)):
            # ignore poses too close to the borders
            x = self.keypoints[i][kp*3+0]
            y = self.keypoints[i][kp*3+1]

            if x < 50 or x > FramePoses.img_w - 50 or y < 50 or y > FramePoses.img_h - 50:
                continue

            # try to associate pose i to previous pose k
            if len(prev_frame_poses) > 0:
                for k in range(0, len(prev_frame_poses[-1])):
                    if prev_frame_poses[-1].track_ids[k] is None or prev_frame_poses[-1].track_next[k] is not None:
                        continue # skip poses with no id (discarded) or already assigned (to other tracks)

                    # calculate distance to keypoint
                    dist = (x - prev_frame_poses[-1].keypoints[k][kp*3+0]) ** 2 + (
                        y - prev_frame_poses[-1].keypoints[k][kp*3+1]) ** 2

                    if dist < 20**2: # assign the id
                        track_id = prev_frame_poses[-1].track_ids[k]
                        self.track_ids[i] = track_id
                        #tracks[track_id]['poses'].append(self.keypoints[i][0*3:8*3] + self.keypoints[i][15*3:19*3])
                        prev_frame_poses[-1].track_next[k] = i
                        self.track_prev[i] = k
                        self.track_len[i] = prev_frame_poses[-1].track_len[k] + 1

            # if the pose was not assigned a track, create a new track
            if self.track_ids[i] is None:
                track_id = FramePoses.next_id
                self.track_ids[i] = track_id
                #tracks.append({'ini': self.frame_num, 'poses':list()})
                #tracks[track_id]['poses'].append(self.keypoints[i][0*3:8*3] + self.keypoints[i][15*3:19*3])
                FramePoses.next_id += 1
                self.track_len[i] = 1

        
        ended_tracks = FramePoses.get_all_ended_tracks(prev_frame_poses)
        tracks += ended_tracks

    def assign_tracks_better(self, active_tracks, tracks):
        kp = 1
        for i in range(0, len(self)):
            # ignore poses too close to the borders
            x = self.keypoints[i][kp*3+0]
            y = self.keypoints[i][kp*3+1]

            if x < 50 or x > FramePoses.img_w - 50 or y < 50 or y > FramePoses.img_h - 50:
                continue

            # try to associate pose i to previous pose k
            if len(active_tracks) > 0:

                # check through all of them and pick the closest one
                dists = np.empty((len(active_tracks)))
                for k, at in enumerate(active_tracks):
                    # calculate distance to keypoint
                    dists[k] = (x - at['poses'][-1][kp*3+0])**2 + (y - at['poses'][-1][kp*3+1])**2

                num_close_tracks = np.sum(dists < 20**2)

                if num_close_tracks == 1:
                    closest_idx = np.argmin(dists)
                    at = active_tracks[closest_idx]
                # if min_dist < 25**2: # assign the id
                    #while at['ini'] + len(at['poses']) < self.frame_num:
                    diff = self.frame_num - (at['ini'] + len(at['poses']))
                    assert diff <= 30
                    if diff > 0:
                        at['poses'].extend(np.zeros((diff,75)))
                    at['poses'].append(self.keypoints[i])
                    self.track_ids[i] = at['id']

            # if the pose was not assigned a track, create a new track
            if self.track_ids[i] is None:
                track_id = FramePoses.next_id
                self.track_ids[i] = track_id
                active_tracks.append({
                    'id': track_id,
                    'ini': self.frame_num,
                    'poses': DynamicArray(np.array(self.keypoints[i])[None,:])
                })
                FramePoses.next_id += 1

        # check for ended/inactice tracks
        tracks += [at for at in active_tracks if at['ini'] + len(at['poses']) <= self.frame_num - 30]
        active_tracks[:] = [at for at in active_tracks if not at['ini'] + len(at['poses']) <= self.frame_num - 30]

    def assign_tracks_fast(self, active_tracks, tracks):
        kp = 1

        keypoints = np.array(self.keypoints)
        w_condition = np.logical_and(keypoints[:,kp*3+0] > 50, keypoints[:,kp*3+0] < FramePoses.img_w - 50)
        h_condition = np.logical_and(keypoints[:,kp*3+1] > 50, keypoints[:,kp*3+1] < FramePoses.img_h - 50)
        keypoints = keypoints[np.logical_and(w_condition, h_condition), :]
        chest_kp = keypoints[:,kp*3+0:kp*3+2]

        active_poses = np.array([at['poses'][-1][kp*3+0:kp*3+2] for at in active_tracks])

        if len(keypoints) > 0 and len(active_poses) > 0:
            # compute distance matrix
            dists  = euclidean_distances(chest_kp, active_poses)

        for i in range(0, len(chest_kp)):
            assigned = False

            if len(active_poses) > 0:
                num_close_tracks = np.sum(dists[i,:] < 20)
                # if num_close_tracks > 5:
                #     print('here')
                #     print(num_close_tracks)
                #     print(dists[i,dists[i,:] < 25])

                if num_close_tracks == 1:
                    closest_idx = np.argmin(dists[i,:])
                    # print('closest: {:d}'.format(closest_idx))

                    at = active_tracks[closest_idx]
                    diff = self.frame_num - (at['ini'] + len(at['poses']))
                    assert diff <= 30
                    if diff > 0:
                        at['poses'].extend(np.zeros((diff,75)))
                    at['poses'].append(keypoints[i])
                    assigned = True

            if not assigned: # create a new track for the pose
                track_id = FramePoses.next_id
                active_tracks.append({
                    'id': track_id,
                    'ini': self.frame_num,
                    'poses': DynamicArray(np.array(keypoints[i])[None,:])
                })
                FramePoses.next_id += 1

        # check for ended/inactive tracks
        tracks += [at for at in active_tracks if at['ini'] + len(at['poses']) <= self.frame_num - 30]
        active_tracks[:] = [at for at in active_tracks if not at['ini'] + len(at['poses']) <= self.frame_num - 30]

    def assign_tracks_hungarian(self, active_tracks, tracks):
    # >>> linear_sum_assignment([[0.5,0.2],[0.2,0.5]])
    # (array([0, 1]), array([1, 0]))
    # >>> linear_sum_assignment([[0.5,0.2],[0.2,0.5],[0.6,0.6]])
    # (array([0, 1]), array([1, 0]))
    # >>> linear_sum_assignment([[0.5,0.2],[0.2,0.5],[0.1,0.1]])
    # (array([0, 2]), array([1, 0]))
        max_diff = 90
        kp = 1
        
        if len(self.keypoints) == 0:
            tracks += [at for at in active_tracks if at['ini'] + len(at['poses']) <= self.frame_num - max_diff]
            active_tracks[:] = [at for at in active_tracks if not at['ini'] + len(at['poses']) <= self.frame_num - max_diff]
            return

        # print(self.frame_num)

        keypoints = np.array(self.keypoints)
        w_condition = np.logical_and(keypoints[:,kp*3+0] > 50, keypoints[:,kp*3+0] < FramePoses.img_w - 50)
        h_condition = np.logical_and(keypoints[:,kp*3+1] > 50, keypoints[:,kp*3+1] < FramePoses.img_h - 50)
        keypoints = keypoints[np.logical_and(w_condition, h_condition), :]
        chest_kp = keypoints[:,kp*3+0:kp*3+2]

        active_poses = np.array([at['poses'][-1][kp*3+0:kp*3+2] for at in active_tracks])

        def create_track(pose):
            track_id = FramePoses.next_id
            active_tracks.append({
                'id': track_id,
                'ini': self.frame_num,
                'poses': DynamicArray(np.array(pose)[None,:])
            })
            FramePoses.next_id += 1

        pose_ids = []
        if len(keypoints) > 0 and len(active_poses) > 0:
            # compute distance matrix
            dists  = euclidean_distances(chest_kp, active_poses)
            hungarian_dists = np.hstack([dists, 30*np.ones((len(chest_kp), len(chest_kp)))])
            pose_ids, track_ids = linear_sum_assignment(hungarian_dists)

            # check the matches
            for pose_id, track_id in zip(pose_ids, track_ids):
                # assign if they are close
                assert hungarian_dists[pose_id, track_id] <= 30

                if track_id < len(active_tracks): # assigned to an added pose
                    at = active_tracks[track_id]
                    diff = self.frame_num - (at['ini'] + len(at['poses']))
                    assert diff <= max_diff
                    if diff > 0:
                        at['poses'].extend(np.zeros((diff,75)))
                    at['poses'].append(keypoints[pose_id])

                else:
                    create_track(keypoints[pose_id])

        # create new tracks for unnasigned poses
        unassigned_poses = np.setdiff1d(range(0,len(chest_kp)), pose_ids)
        for pose_id in unassigned_poses:
            create_track(keypoints[pose_id])

        # check for ended/inactice tracks
        tracks += [at for at in active_tracks if at['ini'] + len(at['poses']) <= self.frame_num - max_diff]
        active_tracks[:] = [at for at in active_tracks if not at['ini'] + len(at['poses']) <= self.frame_num - max_diff]

    def render_poses(self, frame, **kwargs):
        pose_renderer.render_poses(self.keypoints, frame, self.track_ids, **kwargs)
    
    def render_track_info(self, frame):
        pose_renderer.render_track_info(self.keypoints, frame, self.track_ids)

def openpose_keypoints_to_text(args):
    keypoints_path = '/home/jose/data/realvad/pose'
    files = sorted(os.listdir(keypoints_path))
    files = [os.path.join(keypoints_path, f) for f in files]
    fout_name = '/home/jose/data/realvad/pose.txt'
    with open(fout_name, 'w') as fout:
        for fname in tqdm(files):
            frame_poses = FramePoses.from_openpose_json(fname)
            fout.write(str(frame_poses)+'\n')



    

def main(args):
    import pickle
    import track_utils

    my_poses = pickle.load(open('/home/jose/furnace/lared/pose/my_poses.pkl','rb'))
    active_tracks = list()
    tracks = list()

    FramePoses.next_id = 0

    for i in tqdm(range(0, len(my_poses))):
        fp = FramePoses(my_poses[i], frame_num=i)
        fp.assign_tracks_hungarian(active_tracks, tracks)
    tracks += active_tracks
    tracks = sorted(tracks, key=lambda x: x['ini'])
    track_utils.convert_to_upper_body(tracks)
    pickle.dump(tracks, open('unfiltered.pkl', "wb"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tests the class')

    parser.add_argument('-d',
                        '--debug',
                        help="Print lots of debugging statements",
                        action="store_const",
                        dest="loglevel",
                        const=logging.DEBUG,
                        default=logging.WARNING)
    parser.add_argument('-v',
                        '--verbose',
                        help="Be verbose",
                        action="store_const",
                        dest="loglevel",
                        const=logging.INFO)

    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel)
    openpose_keypoints_to_text(args)

