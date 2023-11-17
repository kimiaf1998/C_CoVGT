import json
import os
import os.path as osp
from PIL import Image
import numpy as np
import pandas as pd


def get_difference(list1, list2) -> list:
    return [item for item in list1 if item not in list2]


def find_dividers(lst: list, k: int) -> list:
    if k == len(lst):
        return lst

    section_size = len(lst) // (k + 1)
    divider_indices = [i * section_size for i in range(1, k + 1)]
    divider_elements = [lst[index] for index in divider_indices]

    return divider_elements


def extract_clips_with_keyframes_included(frame_list: list, key_frames: list, num_clips: int,
                                          num_frames_per_clip: int) -> (list, bool):
    frame_count_out = num_clips * num_frames_per_clip

    if frame_count_out >= len(key_frames):
        k = frame_count_out - len(key_frames)
        resp = [i for i in key_frames]
        resp.extend(find_dividers(get_difference(frame_list, key_frames), k))
        resp = sorted(resp)
    else:
        resp = find_dividers(key_frames, frame_count_out)

    # extract output frames values
    # frame_list_out = [np.asarray(Image.open(osp.join(path, frame_no))) for frame_no in resp]
    return np.asarray(resp).reshape(num_clips, num_frames_per_clip)


def sample_videos_clips(video_path: str, ann_path: str, num_clips: int, num_frames_per_clip: int):
    with open(ann_path, 'r') as fp:
        ann_data = json.load(fp)
    videos = []
    qids = []
    for vid_data in ann_data:
        vid_id = vid_data['video_id']
        vid_qid = vid_data['question_id']
        vid_key_frames = list(vid_data['bboxes'].keys())
        vid_frames_dir = os.path.join(video_path, vid_id)
        print(os.path.basename(vid_frames_dir))
        try:
            vid_frames = os.listdir(vid_frames_dir)
            # extract frame numbers
            vid_frames = sorted([os.path.splitext(frame_path.split("-")[1])[0] for frame_path in vid_frames])
            vid_clips = extract_clips_with_keyframes_included(vid_frames, vid_key_frames, num_clips,
                                                              num_frames_per_clip)
            videos.append({vid_qid: vid_clips.tolist()})
            qids.append(vid_qid)
        except FileNotFoundError as e:
            print(f"File not found: {e.filename}")

    return videos, qids


def generate_json(data: list, output_path: str):
    with open(output_path, 'w') as json_file:
        json.dump(data, json_file)


# TODO (sample test from val ds)

if __name__ == '__main__':
    dataset = 'STAR'
    video_dir = f'../../data/{dataset}/frames_24fps/'  # extracted video frames, refer to extract_video.py
    # modes = ['train', 'val', 'test']
    modes = ['train', 'val']
    for mode in modes:
        print("Mode: ", mode)
        ann_path = f'../datasets/{dataset}/{dataset}_{mode}_reformatted.json'  # extracted video frames, refer to extract_video.py

        with open(ann_path, 'r') as fp:
            ann_data = json.load(fp)
        sampled_clips, question_ids = sample_videos_clips(video_dir, ann_path, 8, 4)
        clips_output_path = f'../datasets/{dataset}/clips_{mode}.json'
        videos_output_path = f'../datasets/{dataset}/{mode}.json'

        with open(ann_path, 'r') as fp:
            ann_data = json.load(fp)

        sampled_videos = [data for data in ann_data if data['question_id'] in question_ids]
        # generate_json(sampled_clips, clips_output_path)
        generate_json(sampled_videos, videos_output_path)