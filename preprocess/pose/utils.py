import os
import pickle
import hashlib

from tqdm import tqdm
from .frame_poses import FramePoses

import warnings
import math

import numpy as np
from scipy.interpolate import interp1d
from . import pose_parameters
from ...constants import valid_frame_intvals

def fix_poses(poses, scale):
    poses = poses.copy()

    # extend the arms
    arm_pairs = [(pose_parameters.POSE_UPPER_BODY_PAIRS[i*2], pose_parameters.POSE_UPPER_BODY_PAIRS[i*2+1]) for i in [4,6]]
    for arm_pair in arm_pairs:
        armv = poses[:, arm_pair[1]*3: arm_pair[1]*3+2] - poses[:, arm_pair[0]*3: arm_pair[0]*3+2]
        armv *= 1.15
        poses[:, arm_pair[1]*3: arm_pair[1]*3+2] = poses[:, arm_pair[0]*3: arm_pair[0]*3+2] + armv

    # set parts with 0 confidence to NaN
    zero_conf_idxs = poses[:,2::3] == 0
    poses[:,0::3][zero_conf_idxs] = np.nan
    poses[:,1::3][zero_conf_idxs] = np.nan

    # create the new head kp
    head_points_x = np.hstack([poses[:,i*3+0][:,None] for i in [0,9,10,11,12]])
    head_points_y = np.hstack([poses[:,i*3+1][:,None] for i in [0,9,10,11,12]])
    # with warnings.catch_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        head_mid_x = np.nanmean(head_points_x, axis=1)
        head_mid_y = np.nanmean(head_points_y, axis=1)
    head_mid = np.stack([head_mid_x, head_mid_y], axis=-1)

    # extrapolate the torso to the head point
    # point at mid-section is reference
    v = (poses[:, 1*3: 1*3+2] - poses[:, 8*3: 8*3+2])*[1,-1] # vector from mid-section to neck
    hp = (head_mid - poses[:, 8*3: 8*3+2])*[1,-1]
    u_v = v / np.linalg.norm(v, axis=1)[:,None]
    hp_v = u_v * (hp * u_v).sum(axis=1)[:,None]
    extrap = hp_v*[1,-1] + poses[:, 8*3: 8*3+2]
    # extrap[:,1] -= (60*scale)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        new_head_pt = np.nanmean(np.stack([head_mid, extrap], axis=-1), axis=-1)
    new_head_pt[:,1] -= (50*scale)
    poses[:,0:2] = np.nan_to_num(new_head_pt) # np.nan_to_num(extrap)
    poses[:,2] = np.nan_to_num(new_head_pt.sum(axis=1))

    # remove the rest of the head points
    for i in [9, 10, 11, 12]:
        poses[:, i*3+0] = np.nan
        poses[:, i*3+1] = np.nan
        poses[:, i*3+2] = 0

    # remove the torso point
    poses[:,8*3:8*3+3] = 0

    return poses

def lineseg_dists(p, a, b):
    '''
    Calculates distance between single line and multiple points
    '''
    a = np.array(a)
    b = np.array(b)

    # TODO for you: consider implementing @Eskapp's suggestions
    if not np.any(a - b):
        return np.linalg.norm(p - a, axis=1)

    # normalized tangent vector
    d = np.divide(b - a, np.linalg.norm(b - a))

    # signed parallel distance components
    s = np.dot(a - p, d)
    t = np.dot(p - b, d)

    # clamped parallel distance
    h = np.maximum.reduce([s, t, np.zeros(len(p))])

    # perpendicular distance component, as before
    # note that for the 3D case these will be vectors
    c = np.cross(p - a, d)

    # use hypot for Pythagoras to improve accuracy
    return np.hypot(h, c)

def transform_poses(poses, rect, vsize):
    r_poses = poses - np.tile(np.concatenate([rect[0:2], [0]]), poses.shape[1] // 3)
    scale_factors = [vsize[0] / rect[2], vsize[1] / rect[3], 1]
    res = r_poses * np.tile(scale_factors, poses.shape[1] // 3)
    return res

def get_bone_length(pose, bone, pairs):
    pair = (pairs[bone*2+0], pairs[bone*2+1])
    if pose[pair[0]*3+2] == 0 or pose[pair[1]*3+2] == 0:
        return None # return None if one of the keypoints does not exist
    x0 = pose[pair[0]*3+0]
    y0 = pose[pair[0]*3+1]

    x1 = pose[pair[1]*3+0]
    y1 = pose[pair[1]*3+1]

    dist = np.sqrt((x1-x0)**2 + (y1-y0)**2)
    return dist

def enlarge_rectangle(rect, res=(1920,1080), perc = 0.1, cast_to_int=False):
    extra_w = perc * rect[2]
    extra_h = perc * rect[3]
    tl = [max(0, rect[0] - extra_w), max(0, rect[1] - extra_h)]
    br = [min(res[0], rect[0] + rect[2] + 2*extra_w),
          min(res[1], rect[1] + rect[3] + 2*extra_h)]
    new_rect = [*tl, br[0] - tl[0], br[1] - tl[1]]

    if cast_to_int:
        return [round(elem) for elem in new_rect]

    return new_rect

def get_keypoints_rectangle_corners(keypoints):
    num_keypoints = len(keypoints) // 3

    max_x = max_y = float('-inf')
    min_x = min_y = float('inf')

    for part in range(0, num_keypoints):
        score = keypoints[3*part+2]
        if score == 0:
            continue
        x = keypoints[3*part]
        y = keypoints[3*part+1]

        max_x = max(max_x, x)
        min_x = min(min_x, x)
        max_y = max(max_y, y)
        min_y = min(min_y, y)

    return (min_x, min_y, max_x, max_y)

def get_keypoints_rectangle(keypoints):
    (min_x, min_y, max_x, max_y) = get_keypoints_rectangle_corners(keypoints)
    
    return (min_x, min_y, max_x-min_x, max_y-min_y)

def get_track_rectangle(track):
    x1 = np.empty((len(track)))
    y1 = np.empty((len(track)))
    x2 = np.empty((len(track)))
    y2 = np.empty((len(track)))
    for i in range(0, len(track)):
        x1[i], y1[i], x2[i], y2[i] = get_keypoints_rectangle_corners(track[i,:])

    return int(min(x1)), int(min(y1)), int(max(x2)-min(x1)), int(max(y2)-min(y1))

def tlwh_to_tlbr(rect):
    return (rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3])

def get_track_bbs(track):
    bbs = np.empty((len(track['poses']), 4))
    for i in range(0, track['poses'].shape[0]):
        bbs[i,:] = get_keypoints_rectangle(np.squeeze(track['poses'][i,:]))

    return bbs

def reset_track_ids(tracks):
    for i,track in enumerate(tracks):
        track['id'] = i

def convert_pose_to_upper_body(pose):
    return np.hstack([pose[:,0*3:9*3], pose[:,15*3:19*3]])

def convert_to_upper_body(tracks):
    for t in tracks:
        t['poses'] = np.hstack([t['poses'][:,0*3:9*3], t['poses'][:,15*3:19*3]])

def interp_track(poses):
    zero_block_idx = np.nonzero(np.diff(poses.sum(axis=1) == 0))[0]
    segments = list()

    zero_block_ini = zero_block_idx[0::2]
    zero_block_fin = zero_block_idx[1::2]

    for ini, fin in zip(zero_block_ini, zero_block_fin):
        cols = np.logical_and(poses[ini, :] != 0, poses[fin+1] != 0)
        f = interp1d([ini,fin+1], np.array([poses[ini, cols], poses[fin+1, cols]]), axis=0)
        res = f(np.arange(ini, fin+1))
        poses[ini: fin+1, cols] = res
        segments.append((ini,fin))

    return segments

def interp_tracks(tracks):
    for track in tracks:
        poses = track['poses']
        segments = interp_track(poses)
        track['interp'] = segments
        
def compare_poses(pose1, pose2):
    idx = 1
    return pose1[idx*3+1] == pose2[idx*3+1] and pose1[idx*3+2] == pose2[idx*3+2]


def get_assigned_tracks(tracks, track_assignments):
    ''' Return only tracks in tracks assigned to a pid in track_assignments
        The rest of the tracks were considered invalid in annotation
    '''
    assigned_tracks = list()
    # load the pids
    for track in tracks:
        key = hashlib.sha1(track['poses'].data.tobytes()).hexdigest()
        if key in track_assignments and track_assignments[key] != 'r':
            track['pid'] = track_assignments[key]
            assigned_tracks.append(track)
    return assigned_tracks

def undo_track_interpolation(tracks):
    ''' Reverses the interpolation of the tracks that was done to
        prepare for manual assignment.
    '''
    new_tracks = list()
    for t in tracks:
        last_fin = 0
        t['interp'].append((len(t['poses']),math.inf))
        for seg in t['interp']:
            ini = seg[0]
            fin = seg[1]
            if fin - ini > 15:
                if ini - last_fin > 90:
                    poses = t['poses'][last_fin:ini,:]
                    new_tracks.append({
                        'id': t['id'],
                        'ini': t['ini'] + last_fin,
                        'poses': poses,
                        'pid': t['pid']
                    })
                last_fin = fin
    return new_tracks

def filter_joints_inplace_by_confidence(tracks):
    for track in tracks:
        for i in range (2, track['poses'].shape[1], 3):
            if i == 5: # ignore the neck point
                continue
            idxs = track['poses'][:,i] < 0.5
            track['poses'][idxs,i] = 0

def filter_joints_inplace_by_bone_length(tracks):
    # remove joints if bone too big
    # tuples of (joint1, joint2, max_len)
    to_remove = [
        (0, 8, 225),
        (1, 2, 160),
        (2, 5, 160),
        (3, 3, 160),
        (4, 4, 160),
        (5, 6, 160),
        (6, 7, 160),
        (7, [0,9,10,11,12], 150),
        (8, [9,11], 30),
        (10, [10,12], 30),
        (9, 11, 50),
        (11, 12, 50),
    ] # bone, joint, threshold
    
    for bone, joint, thresh in to_remove:
        for track in tracks:
            track_len = np.empty((len(track['poses'])))
            for i in range (0, track['poses'].shape[0]):
                bone_len = get_bone_length(track['poses'][i,:], bone, pose_parameters.POSE_UPPER_BODY_PAIRS)
                if bone_len is None:
                    continue
                if bone_len > thresh:
                    if type(joint) == int:
                        joint = [joint]
                    for j in joint:
                        track['poses'][i,j*3+2] = 0

def filter_tracks_by_time(tracks):
    new_tracks = list()
    for intval in valid_frame_intvals:
        for track in tracks:
            if track['ini'] >= intval[1] or track['ini'] + len(track['poses']) < intval[0]:
                continue
            prev_ini = track['ini']
            new_ini = max(intval[0], track['ini'])
            new_fin = min(intval[1], track['ini'] + len(track['poses']))

            new_track = dict(track)
            new_track['ini'] = new_ini
            new_track['poses'] = track['poses'][new_ini - prev_ini: new_fin - prev_ini,:]
            new_tracks.append(new_track)
    return new_tracks

def find_track_by_id(tracks, tid):
    for track in tracks:
        if track['id'] == tid:
            return track
    return None

def openpose_keypoints_to_pickle(poses_path, out_path):
    files = sorted(os.listdir(poses_path))
    files = [os.path.join(poses_path, f) for f in files]
    
    all_frame_poses = list()
    for fname in tqdm(files):
        frame_poses = FramePoses.from_openpose_json(fname)
        all_frame_poses.append(frame_poses.keypoints)

    pickle.dump(all_frame_poses, open(out_path, 'wb'))

def openpose_keypoints_to_text(poses_path, out_path):
    ''' Faster than the pickle version
        Will turn the (framewise) output from openpose (applied to a video)
        into a single large pickle file which is faster to load.
    '''
    files = sorted(os.listdir(poses_path))
    files = [os.path.join(poses_path, f) for f in files]
    with open(out_path, 'w') as fout:
        for fname in tqdm(files):
            frame_poses = FramePoses.from_openpose_json(fname)
            fout.write(str(frame_poses)+'\n')