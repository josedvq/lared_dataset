import pickle
import argparse
import logging
import hashlib
import os

import numpy as np
import cv2

from lared_dataset.preprocess.pose import pose_parameters
import pose_renderer

keymaps = {
    'ubuntu': {
        65362: 'arrow_top',
        65364: 'arrow_bottom',
        65361: 'arrow_left',
        65363: 'arrow_right',
        93: 'right_bracket',
        91: 'left_bracket',
        27: 'escape',
        32: 'space',
        65288: 'backspace',
        65535: 'supr'
    }
}

class TrackAnnotator:

    def __init__(self, cap, tracks, track_assignment=None, log_path=None):
        '''
        assumes that tracks is sorted
        '''
        # opencv
        self.cap = cap
        self.num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.input_res = (1920,1080)

        # tracks
        self.load_tracks(tracks)
        self.frame = None
        self.out_frame = None
        self.curr_frame = None # frame num
        self.active_tracks = list()
        
        # person patches
        self.person_patches = dict()#queue.Queue()
        self.log_enabled = False
        if log_path is not None:
            self.log_enabled = True
            self.log_path = log_path
            self.person_patches_path   = os.path.join(log_path, 'person_patches')
            if not os.path.exists(self.person_patches_path):
                os.mkdir(self.person_patches_path)

        # track assignment
        self.load_track_assignment(track_assignment)
        self.track_assignment_path = track_assignment

        # annotation state
        self.associated_poses = list()
        self.nonassociated_tracks = list()
        self.curr_track= 0
        self.tracks_idx = 0

        # user-input related
        self.mode = 'annot'
        self.input_keys = ''
        self.keymap = keymaps['ubuntu']

        # output
        self.output_res = (960, 540)
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        # track-based nav
        self.this_track = 0

    def load_tracks(self, tracks):
        if type(tracks) == str:
            self.load_tracks_from_file(tracks)
            
        elif type(tracks) == list:
            self.tracks = sorted(tracks, key=lambda x: x['ini'])
        else:
            raise 'unknown type for tracks'

    def load_tracks_from_file(self, tracks):
        try:
            print('loading tracks')
            t = pickle.load(open(tracks, "rb"))
            self.tracks = sorted(t, key=lambda x: x['ini'])
            self.tracks_file = tracks
        except:
            raise 'error loading tracks. no tracks loaded.'

    def load_track_assignment(self, track_assignment):
        if track_assignment is None:
            self.track_assignment = dict()
        elif type(track_assignment) == dict:
            self.track_assignment = track_assignment
        elif type(track_assignment) == str:
            try:
                print('loading track assignment')
                self.track_assignment = pickle.load(open(track_assignment, "rb"))
                self.track_assignment_path = track_assignment
            except:
                self.track_assignment = dict()
        else:
            raise 'unknown type for track_assignment'

        # load the pids
        for track in self.tracks:
            key = self.get_assignment_key(track)
            if key in self.track_assignment:
                track['pid'] = self.track_assignment[key]
            else:
                track['pid'] = None

    def goto_track_by_id(self, id):
        for i, t in enumerate(self.tracks):
            if t['id'] == id:
                print(t['id'])
                t = self.tracks[i]
                self.load_frame(t['ini'])
                self.set_curr_track_by_id(t['id'])
                self.this_track = id
                return t
        return False

    def set_curr_track_by_id(self, id):
        for i, track in enumerate(self.active_tracks):
            if track['id'] == id:
                self.curr_track = i
                return
        raise 'did not find track'

    def save_track_assignment(self):
        pickle.dump(self.track_assignment, open(self.track_assignment_path, "wb"))
        print('saved file.\n')

    def make_next_frame_annot(self):
        self.out_frame = self.frame.copy()
        # draw the associated poses in green
        if len(self.associated_poses) > 0:
            colors = [0, 255, 0] * int(len(self.associated_poses[0]) // 3)
            pose_renderer.render_poses_simple(self.associated_poses, self.out_frame, pose_parameters.POSE_UPPER_BODY_PAIRS, colors)
            pose_renderer.render_track_info(self.associated_poses, self.out_frame, self.associated_poses_ids)

        # draw the nonassociated poses/tracks
        if len(self.nonassociated_tracks) > 0:
            nonassociated_poses = [p['poses'][self.curr_frame - p['ini']] for p in self.nonassociated_tracks]
            nonassociated_poses_ids = [p['pid'] for p in self.nonassociated_tracks]
            ids = [1] * len(nonassociated_poses)
            # colors = [255, 255, 255] * int(len(nonassociated_poses[0]) // 3)
            colors = [0, 255, 0] * int(len(nonassociated_poses[0]) // 3)
            pose_renderer.render_poses_simple(nonassociated_poses, self.out_frame, pose_parameters.POSE_UPPER_BODY_PAIRS, colors)
            pose_renderer.render_track_info(nonassociated_poses, self.out_frame, nonassociated_poses_ids)

            # draw the queried pose
            query_track = self.nonassociated_tracks[self.curr_track]
            t = self.curr_frame - query_track['ini']

            # draw the active pose in red
            # colors = [0, 0, 255] * int(len(query_track['poses'][t]) // 3)
            pose_renderer.render_poses_simple([query_track['poses'][t]], self.out_frame, pose_parameters.POSE_UPPER_BODY_PAIRS, colors)

    def update_poses(self):
        # update the associated_poses and non_associated tracks from the active tracks
        self.associated_poses = list()
        self.associated_poses_ids = list()
        self.nonassociated_tracks = list()
        
        for at in self.active_tracks:
            t = self.curr_frame - at['ini']
            if self.mode == 'edit': # in edit mode all tracks are non-associated
                self.nonassociated_tracks.append(at)
            else:
                if at['pid'] is not None:
                    if at['pid'] != 'r':
                        # if t > len(at['poses']):
                        self.associated_poses.append(at['poses'][t])
                        self.associated_poses_ids.append(at['pid'])
                else:
                    self.nonassociated_tracks.append(at)

        if self.curr_track > len(self.nonassociated_tracks):
            self.curr_track = 0
        print((len(self.active_tracks), len(self.nonassociated_tracks), len(self.associated_poses)))

    def load_frame(self, frame_num):
        assert frame_num < self.num_frames

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num+1) # frame_num is zero-based
        ret, self.frame = self.cap.read()

        # get the active tracks
        self.active_tracks = list()
        self.tracks_idx = 0
        for i,t in enumerate(self.tracks):
            if t['ini'] <= frame_num and t['ini'] + len(t['poses']) > frame_num:
                self.active_tracks.append(t)
            if t['ini'] <= frame_num:
                self.tracks_idx = i+1
        #self.active_tracks = [t for t in self.tracks if t['ini'] <= frame_num and t['ini'] + len(t['poses']) > frame_num]
        assert self.tracks[self.tracks_idx]['ini'] >= frame_num+1, '{:d} !>= {:d}'.format(self.tracks[self.tracks_idx]['ini'], frame_num+1)
        print((self.tracks_idx, len(self.tracks), self.tracks[self.tracks_idx]['ini'], self.curr_frame))
        self.curr_frame = frame_num
        self.curr_track= 0

        self.update_poses()

    def load_next_frame(self):
        print('loading next frame..')
        
        # gather the tracks active in the current frame
        if self.tracks_idx < len(self.tracks) and self.curr_frame + 1 < self.num_frames:
            ret, self.frame = self.cap.read()
            self.curr_frame += 1
            self.curr_track= 0

            # remove active tracks that are ended
            self.active_tracks = [t for t in self.active_tracks if t['ini'] + len(t['poses']) > self.curr_frame]

            # update the active tracks with the tracks starting in the current frame
            print((self.tracks_idx, len(self.tracks), self.tracks[self.tracks_idx]['ini'], self.curr_frame))
            # assert self.tracks[self.tracks_idx]['ini'] >= self.curr_frame
            while self.tracks_idx < len(self.tracks) and self.tracks[self.tracks_idx]['ini'] == self.curr_frame:
                self.active_tracks.append(self.tracks[self.tracks_idx])
                self.tracks_idx += 1
                print((self.tracks_idx, len(self.tracks), self.tracks[self.tracks_idx]['ini'], self.curr_frame))
            self.update_poses()
        else:
            raise ValueError

    def get_assignment_key(self, track):
        return hashlib.sha1(track['poses'].data.tobytes()).hexdigest()

    def assign_track(self, track, annotation):
        key = self.get_assignment_key(track)
        track['pid'] = annotation
        self.track_assignment[key] = annotation

        # save the patch
        if self.log_enabled:
            self.process_person_patch(track, annotation)

    def process_person_patch(self, track, annotation):
        t = self.curr_frame - track['ini']
        rect = pose_renderer.get_keypoints_rectangle(track['poses'][t])
        if type(annotation) == int:
            if annotation not in self.person_patches:
                self.person_patches[annotation] = list()
            # keep the len to 4
            if len(self.person_patches[annotation]) == 4:
                self.person_patches[annotation] = self.person_patches[annotation][1:]

            patch = self.frame[
                max(0, int(rect[1]-0.2*rect[3])): min(self.input_res[1], int(rect[1]+1.2*rect[3])), 
                max(0, int(rect[0]-0.2*rect[2])): min(self.input_res[0], int(rect[0]+1.2*rect[2])),
                :]
            self.person_patches[annotation].append(patch)
            self.output_person_patch(annotation)

    def output_person_patch(self, pid):
        patches = self.person_patches[pid]
        img = np.zeros((800,600,3), np.uint8)

        k = 0
        for i in range(0,2):
            for j in range(0,2):
                if k >= len(patches):
                    continue
                patch = patches[k]
                print(patch.shape)
                ratio = 300 / patch.shape[1]
                dims = (300, int(ratio * patch.shape[0]))
                patch = cv2.resize(patch, dims)
                img[j*400: min((j+1)*400, j*400+patch.shape[0]), i*300: (i+1)*300,:] = patch[:min(patch.shape[0], 400), 0:300 ,:]
                k += 1

        img_path = os.path.join(self.person_patches_path, '{:02d}.jpg'.format(pid))
        cv2.imwrite(img_path, img)

    def associate_curr_track(self):
        # process input
        annotation = None
        if self.input_keys.isdigit():
            annotation = int(self.input_keys)
        elif self.input_keys == 'r' or self.input_keys == 'f':
            annotation = self.input_keys
        else:
            print('invalid annotation')
            self.input_keys = ''
            return

        self.input_keys = ''

        self.assign_track(self.nonassociated_tracks[self.curr_track], annotation)
        
        self.update_poses()
        if self.curr_track < len(self.nonassociated_tracks)-1:
            self.curr_track += 1
        else:
            if len(self.nonassociated_tracks) > 0:
                self.curr_track = 0
            else: # if there are no more tracks in the frame
                self.go_to_next_annot()

    def go_to_next_annot(self):
        while len(self.nonassociated_tracks) == 0:
            self.load_next_frame()

    def fast_annot(self):
        # auto remove all tracks until one associated one dissapears
        a = len(self.associated_poses)
        while len(self.associated_poses) == a:
            self.load_next_frame()
            for nat in self.nonassociated_tracks:
                self.assign_track(nat, 'r')

    def print_track_info(self, curr_track):
        # curr_track = self.nonassociated_tracks[self.curr_track]
        t = self.curr_frame - curr_track['ini']
        
        coords = curr_track['poses'][t][3:5]
        print(coords)
        print('curr track ({:d}): {:d} - {:d} : x: {:d} y:{:d}'.format(
            curr_track['id'],
            curr_track['ini'], 
            curr_track['ini'] + len(curr_track['poses']),
            int(coords[0]),
            int(coords[1])
        ))

        confidences = curr_track['poses'][t][2::3]
        body_parts = ['nose', 'neck', 'Rshoulder', 'Relbow', 'Rwrist', 'Lshoulder', 'Lelbow', 'Lwrist', 'midhip', 'Reye', 'Leye', 'Rear', 'Lear']
        out_str = ''
        for bp, c in zip(body_parts, confidences):
            out_str += '{:s}={:.2f} '.format(bp,c)
        print(out_str)

    def process_input_annot(self, k):
        if k in range(48,58):
            self.input_keys += str(k-48)
        elif k == 99:
            self.fast_annot()
        elif k in range(97,122):
            self.input_keys += str(chr(k))
        elif k in self.keymap and self.keymap[k] == 'backspace':
            self.input_keys = self.input_keys[:-1]
        elif k in self.keymap and self.keymap[k] == 'arrow_top':
            if self.curr_track < len(self.nonassociated_tracks) -1:
                self.curr_track += 1
                self.print_track_info(self.nonassociated_tracks[self.curr_track])
        elif k in self.keymap and self.keymap[k] == 'arrow_bottom':
            if self.curr_track > 0:
                self.curr_track -= 1
                self.print_track_info(self.nonassociated_tracks[self.curr_track])
        elif k in self.keymap and self.keymap[k] == 'arrow_left':
            self.load_frame(self.curr_frame-1)
        elif k in self.keymap and self.keymap[k] == 'arrow_right':
            self.load_next_frame()
        elif k == 91: # go 10s back
            self.load_frame(max(0, self.curr_frame - 300))
        elif k == 93: # skip 10s forward
            self.load_frame(min(self.num_frames-1, self.curr_frame + 300))
        elif k == 45: # go to prev track
            t = self.goto_track_by_id(self.this_track-1)
            if t == False:
                print('could not load track. ids missing?')
            else:
                self.print_track_info(t)
        elif k == 61: # go to next track
            t = self.goto_track_by_id(self.this_track+1)
            if t == False:
                print('could not load track. ids missing?')
            else:
                self.print_track_info(t)
        elif k == 39: # go to begin of track
            t = self.active_tracks[self.curr_track]
            self.load_frame(t['ini'])
            self.set_curr_track_by_id(t['id'])
        elif k == 92: # go to end of track
            t = self.active_tracks[self.curr_track]
            self.load_frame(t['ini']+len(t['poses'])-1)
            self.set_curr_track_by_id(t['id'])
        elif k in [13, 65421] : # enter
            self.associate_curr_track()
        else:
            print('invalid input {:d}'.format(k))

    def process_input_goto(self, k):
        if k in range(48,58):
            self.input_keys += str(k-48)
        elif k in self.keymap and self.keymap[k] == 'backspace':
            self.input_keys = self.input_keys[:-1]
        elif k in [13, 65421] : # enter
            self.load_frame(int(self.input_keys))
            self.input_keys = ''
        else:
            print('invalid input {:d}'.format(k))

    def process_input_goto_track(self, k):
        if k in range(48,58):
            self.input_keys += str(k-48)
        elif k in self.keymap and self.keymap[k] == 'backspace':
            self.input_keys = self.input_keys[:-1]
        elif k in [13, 65421] : # enter
            t = self.goto_track_by_id(int(self.input_keys))
            if not t:
                print('could not find track')
            else:
                self.print_track_info(t)
            self.input_keys = ''
        else:
            print('invalid input {:d}'.format(k))

    def process_input(self, k):
        # 'alpha': {97 + c: chr(97 + c) for c in range(0, 26)},
        # 'numeric': {48 + i: i for i in range(0, 10)},
        # print(k)
        print('input {:d}'.format(k))
        if k == 97: # a = annot
            # annot mode
            self.mode = 'annot'
            self.input_keys = ''
            self.update_poses()
        elif k == 116: # g = goto
            self.mode = 'goto'
            self.input_keys = ''
        elif k == 121: # 
            self.mode = 'goto_track'
            self.input_keys = ''
        elif k == 101: # e = edit
            self.mode = 'edit'
            self.input_keys = ''
            self.update_poses()
        elif k == 102: # f = forward until annot
            self.go_to_next_annot()
        elif k == 111: # o = reload
            self.load_tracks_from_file(self.tracks_file)
            self.load_frame(self.curr_frame)
        elif k == 115: # s = save
            self.save_track_assignment()
        elif k == 113: # q = quit
            exit(0)
        else:
            if self.mode == 'annot':
                self.process_input_annot(k)
            elif self.mode == 'goto':
                self.process_input_goto(k)
            elif self.mode == 'goto_track':
                self.process_input_goto_track(k)
            elif self.mode == 'edit':
                self.process_input_annot(k)

            if k in range(97,122):
                print(k)

    def make_next_frame(self):
        if self.mode == 'annot':
            self.make_next_frame_annot()
        elif self.mode == 'goto':
            self.make_next_frame_annot()
        elif self.mode == 'goto_track':
            self.make_next_frame_annot()
        elif self.mode == 'edit':
            self.make_next_frame_annot()

        # draw the current input
        cv2.putText(self.out_frame, '{:s}({:d})>: {:s}'.format(self.mode, self.curr_frame, self.input_keys), (0, self.input_res[1]-5), self.font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    def start_annotation(self):
        self.load_frame(0)

        while True:
            # make next frame using state
            self.make_next_frame()

            # display the frame and ask for input
            f = cv2.resize(self.out_frame, self.output_res)
            cv2.imshow('frame', f)
            k = cv2.waitKeyEx()

            # update the state using the input
            self.process_input(k)
            
def main(args):
    video_path = '/home/jose/data/realvad/realvad.mp4'
    
    cap = cv2.VideoCapture(video_path)

    tracks = './data/tracks/realvad_final.pkl'
    
    # assign = './data/assignments/realvad.pkl'

    save_path = 'savefolder'
    annot = TrackAnnotator(cap, tracks, None)
    annot.start_annotation()


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