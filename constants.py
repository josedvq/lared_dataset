import os
import math

dataset_path = '/mnt/e/data/lared'

raw_data_path = os.path.join(dataset_path, 'raw')
raw_pose_path = os.path.join(raw_data_path, 'pose')
raw_accel_path = os.path.join(raw_data_path, 'accel')
raw_videos_path = os.path.join(raw_data_path, 'videos')

processed_data_path = os.path.join(dataset_path, 'processed')
processed_audio_path = os.path.join(processed_data_path, 'audio')
processed_pose_path = os.path.join(processed_data_path, 'pose')
processed_accel_path = os.path.join(processed_data_path, 'accel')
processed_videos_path = os.path.join(processed_data_path, 'videos')

annotations_path = os.path.join(dataset_path, 'annotations')
vad_path = os.path.join(annotations_path, 'vad')
poses_path = os.path.join(annotations_path, 'pose')
laughter_annot_path = os.path.join(annotations_path, 'laughter')

examples_path = os.path.join(dataset_path, 'examples.pkl')

# valid sections of the data (in frames)
# these segments contain mingling interaction
# used for filtering pose tracks
valid_frame_intvals = [
    (50949, 73426),
    (99800, math.inf)
]

balloon_pop_1_video_frame = 23030 # to 
balloon_pop_1_accel_frame = 45977 + 19/34

balloon_pop_2_video_frame = 74844
balloon_pop_2_accel_frame = 47706 + 23/28

balloon_pop_3_video_frame = 166836.5
balloon_pop_3_accel_frame = 50776 + 30.5/32