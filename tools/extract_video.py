# ====================================================
# @Time    : 15/4/21 12:38 PM
# @Author  : Xiao Junbin
# @Email   : junbin@comp.nus.edu.sg
# @File    : extract_video.py.py
# ====================================================
import os
import os.path as osp
import shutil
import subprocess
import pandas as pd
import json
import sys
sys.path.insert(0, '../')
from util import load_file

# def load_file(filename):
#     with open(filename, 'r') as fp:
#         data = json.load(fp)
#     return data

def get_video_list(filename, out_file):
    data = load_file(filename)
    video_ids = list(data['video_id'])
    video_ids = list(set(video_ids))
    # video_ids = os.listdir(filename)
    # video_ids = sorted(video_ids)
    print(len(video_ids))
    with open(out_file, 'w') as fp:
        json.dump(video_ids, fp, indent=4)
    return video_ids


def extract_frame(video, dst):
    """
    Extract frames from a video at 6 fps using FFmpeg.

    Removes the existing destination directory if present, then saves the extracted frames
    as high-quality JPEGs in the specified directory.

    Parameters:
    - video (str): Path to the input video.
    - dst (str): Directory to save extracted frames.

    Requires:
    - FFmpeg in system's PATH.

    Returns:
    - None
    """
    
    with open(os.devnull, 'w') as ffmpeg_log:
        if os.path.exists(dst):
            # print(" cleanup: "+dst+"/")
            shutil.rmtree(dst)
        os.makedirs(dst)
        print("dst: ", dst)
        print("video: ", video)
        video2frame_cmd = [
            "ffmpeg",
            '-y',
            '-i', video,
            '-r', "6", # 6 frames per second
            # '-vf', "scale=400:300",
            '-qscale:v', "2",
            '{0}/%06d.jpg'.format(dst)
        ]
        subprocess.call(video2frame_cmd)


def extract_videos(raw_dir, vlist, frame_dir, map_vid=None):
    """
    Extract frames from videos listed in vlist.

    For each video ID in vlist, frames are extracted using extract_frame.
    Optionally, video IDs can be mapped using the map_vid dictionary.

    Parameters:
    - raw_dir (str): Directory with raw videos.
    - vlist (list): Video IDs to process.
    - frame_dir (str): Directory to save extracted frames.
    - map_vid (dict, optional): Dictionary to map video IDs to their saved locations.

    Returns:
    - None
    """
    vnum = len(vlist)
    for id, vid in enumerate(vlist):
        # if id <= 400: continue
        # if id > 400: break
        vid = str(vid)
        if map_vid != None:
            video = osp.join(raw_dir, f'{map_vid[vid]}.mp4')
        else:
            video = osp.join(raw_dir, f'{vid}.mp4')
        dst = osp.join(frame_dir, vid)
        if not osp.exists(video):
            print(video)
        extract_frame(video, dst)
        if id % 20 == 0:
            print('{}/{}'.format(id, vnum))


def main():
    dataset = 'STAR'
    video_dir = f'../../data/{dataset}/'
    raw_dir = osp.join(video_dir, 'videos/')
    frame_dir = osp.join(video_dir, 'frames/')
    anno_dir = f'../datasets/{dataset}/'
    if dataset is not 'STAR':
        vlist_file = osp.join(anno_dir, 'vlist.json')
        map_file = osp.join(anno_dir, 'map_vid_vidorID.json')
        if not osp.exists(vlist_file):
            dset = 'test' #train/val
            qa_file = osp.join(anno_dir, f'{dset}.csv')
            vlist_file = osp.join(anno_dir, f'vlist_{dset}.json')
            vlist = get_video_list(qa_file, vlist_file)
        else:
            vlist = load_file(vlist_file)
        map_vid = load_file(map_file)
    else:
        # TODO
        vlist = os.listdir(raw_dir)
        # Extract file names without suffix
        vlist = [os.path.splitext(file)[0] for file in vlist]
        map_vid = None
    extract_videos(raw_dir, vlist, frame_dir, map_vid=map_vid)


if __name__ == "__main__":
    main()
