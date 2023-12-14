"""
Script to visualize the bounding boxes for a given question ID and video ID.
"""
from moviepy.editor import VideoFileClip
import cv2
import pandas as pd
import os
import shutil

#############################################################################
# TODO: update these parameters
#############################################################################
# question_id = "Interaction_T1_4"
# video_id = "TJZ0P"
question_id = "Interaction_T1_9"
video_id = "DUZDL"
path_to_videos = "./data/star/Charades_v1_480"
# Output directory. The frames with the bounding boxes will be saved here
output_dir = "./data/dev/star/frames_with_bboxes/"
#############################################################################

def draw_bounding_boxes(question_id, video_id, path_to_videos, output_dir, convert_to_3_fps=False):
    # Within the output directory, create a directory for the video
    write_dir = os.path.join(output_dir, video_id)
    if os.path.exists(write_dir):
        # If the directory already exists, remove its contents
        shutil.rmtree(write_dir)
        print(f"Directory '{write_dir}' already exists. Cleared its contents.")
    
    # Create a new directory
    os.makedirs(write_dir)
    print(f"Directory '{write_dir}' created.")
    
    # Path to the input video
    video_path = f'{path_to_videos}/{video_id}.mp4'

    # Load the video clip
    video_clip = VideoFileClip(video_path)

    # Get the frames and the bounding boxes
    df = pd.read_json('./C_CoVGT/datasets/star/STAR_train_reformatted.json')
    row = df[df['question_id'] == question_id].iloc[0]
    assert video_id == row["video_id"]

    frames = list(video_clip.iter_frames())
    for frame, bbox in row["bboxes"].items():
        frame_num = int(frame)
        # Fetch the frame by frame number directly
        extracted_frame = frames[frame_num - 1] # subtract 1 because frames is 0-indexed

        if bbox:
            # Draw the bounding box on the frame
            start_point = (int(bbox[0]), int(bbox[1]))
            end_point = (int(bbox[2]), int(bbox[3]))
            color = (0, 255, 0)  # Green color
            thickness = 2
            extracted_frame = cv2.rectangle(extracted_frame, start_point, end_point, color, thickness)

        # Save the frame with the bounding box
        extracted_frame_filename = f'{video_id}_frame_{frame_num}.jpg'
        output_path = os.path.join(write_dir, extracted_frame_filename)
        cv2.imwrite(output_path, extracted_frame)

        print(f"Bounding box drawn on Frame {frame_num} and saved as {extracted_frame_filename}")
 
    # Close the video clip
    video_clip.close()

    question = row["question"]
    answer = row["answer"]
    print("------------")
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print(f"Bounding boxes are surrounding the answer: {answer}")

draw_bounding_boxes(question_id, video_id, path_to_videos, output_dir)
