"""
Script to read in videos in mp4 format and determine their original fps.
A mapping of (video_id -> fps) is written to OUTPUT_FILE_PATH.
Example contents: {"NB0IB": 24.5, "CUONA": 30.01, ...}
"""

from moviepy.editor import VideoFileClip
import os
import pandas as pd
import json

directory = "./data/star/Charades_v1_480/"
OUTPUT_FILE_PATH = "./C_CoVGT/datasets/star/vid_fps_mapping.json"

video_id_to_fps = {}
for filename in os.listdir(directory):
    if filename.endswith('.mp4'):
        video_id = filename.split(".mp4")[0]
        filepath = os.path.join(directory, filename)
        clip = VideoFileClip(filepath)
        fps = clip.fps
        video_id_to_fps[video_id] = fps

# Writing the dictionary to a JSON file
with open(OUTPUT_FILE_PATH, 'w') as file:
    json.dump(video_id_to_fps, file)
