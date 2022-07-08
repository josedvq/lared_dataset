import math
import argparse
import logging

import cv2
import numpy as np
from tqdm import tqdm

from . import pose_parameters

def get_keypoints_rectangle(keypoints):
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
    
    return (min_x, min_y, max_x-min_x, max_y-min_y)

def draw_lines(keypoints, frame, pairs, colors, thickness, threshold=0):
    num_keypoints = len(keypoints) // 3
    if len(colors) == 3:
        colors = np.repeat(colors, num_keypoints).tolist()

    for pair in range(0, len(pairs), 2):
        index1 = pairs[pair] * 3
        index2 = pairs[pair+1]*3

        if keypoints[index1+2] > threshold and keypoints[index2+2] > threshold:
            thickness_line_scaled = int(round(thickness))

            color_index = pairs[pair+1]*3
            color = (colors[color_index+2],
                    colors[color_index+1],
                    colors[color_index+0])

            keypoint1 = (int(round(keypoints[index1])), int(round(
                keypoints[index1+1])))

            keypoint2 = (int(round(keypoints[index2])),
                        int(round(keypoints[index2+1])))

            cv2.line(frame, keypoint1, keypoint2,
                    color, thickness_line_scaled)

def draw_circles(keypoints, frame, colors, radius, thickness, threshold=0):
    num_keypoints = len(keypoints) // 3
    if len(colors) == 3:
        colors = np.repeat(colors, num_keypoints)

    for part in range(0, num_keypoints):
        kp_index = part * 3
        if keypoints[kp_index + 2] > threshold:
            radius_scaled = int(round(radius))
            thickness_circle_scaled = int(round(thickness))

            color_index = part * 3
            color = (colors[color_index+2],
                    colors[color_index+1],
                    colors[color_index+0])

            center = (int(round(keypoints[kp_index])), int(round(
                keypoints[kp_index+1])))
            cv2.circle(frame, center, radius_scaled,
                    color, thickness_circle_scaled)

def draw_semitransparent_circles(frame, keypoints, radius, colors=pose_parameters.POSE_UPPER_BODY_COLORS, threshold=0):
    num_keypoints = len(keypoints) // 3

    for part in range(0, num_keypoints):
        kp_index = part * 3
        if keypoints[kp_index + 2] > threshold:
            overlay = frame.copy()

            center = (int(round(keypoints[kp_index])), int(round(
                keypoints[kp_index+1])))
            
            color_index = part * 3
            color = (colors[color_index+2],
                    colors[color_index+1],
                    colors[color_index+0])

            cv2.circle(overlay, center, radius[part], color, -1)  # A filled rectangle

            alpha = 0.2  # Transparency factor.

            # Following line overlays transparent rectangle over the image
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    return frame

def render_poses(poses, frame, track_ids=None, pairs=None, colors=None, thickness_circle_ratio=1/150, thickness_line_ratio_wrt_circle=0.75, pose_scales=1, threshold=0, draw_track_info=False):
    '''
    '''
    width = frame.shape[1]
    height = frame.shape[0]
    area = width*height
    
    if track_ids is None:
        track_ids = [None] * len(poses)

    for i,person_keypoints in enumerate(poses):

        assert len(person_keypoints) % 3 == 0

        if not np.any(person_keypoints):
            continue

        if track_ids[i] is not None:
            # draw openpose pose
            person_rectangle = get_keypoints_rectangle(person_keypoints)
            rectangle_area = person_rectangle[2] * person_rectangle[3]

            if rectangle_area > 0:
                ratio_areas = min(
                    1, max(person_rectangle[2]/width, person_rectangle[3]/height))
            else:
                continue

            thickness_ratio = max(
                round(math.sqrt(area) * thickness_circle_ratio * ratio_areas), 1)

            if ratio_areas > 0.05:
                thickness_circle = max(1, thickness_ratio)
            else:
                thickness_circle = 1

            thickness_line = max(
                1, round(thickness_ratio * thickness_line_ratio_wrt_circle))
            radius = thickness_ratio / 2

            # draw lines
            draw_lines(person_keypoints, frame, pairs, colors, thickness_line, threshold)

            # draw circles
            draw_circles(person_keypoints, frame, colors, radius, thickness_circle, threshold)

        else:
            # draw simple skeleton
            lines_color = [255, 255, 255] * int(len(person_keypoints) // 3)
            draw_lines(person_keypoints, frame, pairs, lines_color, 2, threshold)

def render_poses_simple(poses, frame, pairs=pose_parameters.POSE_UPPER_BODY_PAIRS, colors=pose_parameters.POSE_UPPER_BODY_COLORS, radius_circle=4, thickness_line=2):
    '''
    '''
    for i,person_keypoints in enumerate(poses):

        assert len(person_keypoints) % 3 == 0

        # draw lines
        draw_lines(person_keypoints, frame, pairs, colors, thickness_line, 0)

        # draw circles
        if radius_circle != -1:
            draw_circles(person_keypoints, frame, colors, radius_circle, -1, 0)


def render_track_info(poses, frame, track_ids):
    assert len(poses) == len(track_ids)
    kp = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_color = (255, 255, 255)
    font_thickness = 2
    for i, person_keypoints in enumerate(poses):

        if track_ids[i] is None:
            continue # skip poses not assigned to a track

        bottom_left = (int(person_keypoints[kp*3+0] - 5), int(person_keypoints[kp*3+1] + 5))
        rect_bottom_left = (bottom_left[0]-3, bottom_left[1]+3)
        rect_top_right = (bottom_left[0]+30, bottom_left[1]-22)

        cv2.rectangle(frame, rect_bottom_left, rect_top_right, (0,0,255), -1)
        cv2.putText(frame, '{:s}'.format(str(track_ids[i])), bottom_left, font, font_scale, font_color, font_thickness)

def render_tracks(cap, num_frames, tracks, vout, bbs=None):
    '''
    assumes that tracks is sorted
    '''
    active_tracks = list()
    tracks_idx = 0
    ini = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

    for i in tqdm(range(ini, ini+num_frames)):
        # remove active tracks that are ended
        active_tracks = [t for t in active_tracks if not i >= t['ini'] + len(t['poses'])]

        # update the active tracks with the tracks starting in the current frame
        while tracks_idx < len(tracks) and tracks[tracks_idx]['ini'] == i:
            active_tracks.append(tracks[tracks_idx])
            tracks_idx += 1

        # gather the poses for the current frame
        fps = list()
        ftids = list()
        for at in active_tracks:
            t = i - at['ini']
            fps.append(at['poses'][t])
            ftids.append(at['id'])

        # draw the poses
        ret, frame = cap.read()
        render_poses(fps, frame, ftids, pose_parameters.POSE_UPPER_BODY_PAIRS, pose_parameters.POSE_UPPER_BODY_COLORS)
        render_track_info(fps, frame, ftids)
        vout.write(frame)

def main(args):
    # load frame
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tests the MILES classifier.')

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
    main(args)
