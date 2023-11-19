"""
Script to visualize the bounding boxes for a given question ID and video ID.
"""
# from moviepy.editor import VideoFileClip
import cv2
import pandas as pd
import os

#############################################################################
# TODO: update these parameters
#############################################################################
# question_id = "Interaction_T1_4"
# video_id = "TJZ0P"
question_id = "Interaction_T1_5590"
video_id = "OJ4IE"
path_to_videos = "../../data/star"
# Output directory. The frames with the bounding boxes will be saved here
output_dir = "../../data/star/dev"
#############################################################################

def draw_bounding_boxes(question_id, video_id, path_to_videos, output_dir):
    # Within the output directory, create a directory for the video
    write_dir = os.path.join(output_dir, video_id)
    os.makedirs(write_dir, exist_ok=True)
    
    # Path to the input video
    video_path = f'{path_to_videos}/{video_id}'

    # Load the video clip
    # video_clip = VideoFileClip(video_path)

    # Get the frames and the bounding boxes
    df = pd.read_json('../datasets/star/train_updated_frame_number.json')
    row = df[df['question_id'] == question_id].iloc[0]
    assert video_id == row["video_id"]
    for frame, bbox in row["bboxes"].items():
        frame_num = int(frame)
        print("frame num:", frame_num)
        # Calculate the time corresponding to the frame number at 3 fps
        frame_time = frame_num / 3

        # Save the extracted frame as an image file
        # extracted_frame_filename = f'{video_id}_frame_{frame_num}.jpg'
        # output_path = os.path.join(write_dir, extracted_frame_filename)
        output_path = os.path.join(write_dir, f'{frame_num:06d}.png')
        # video_clip.save_frame(output_path, t=frame_time)
        #
        print(f"Frame at {frame_num} seconds extracted and saved as {os.path.join(video_path, f'{frame_num:06d}')}.png")

        # Load the frame
        image = cv2.imread(os.path.join(video_path, f'{frame_num:06d}.png'))

        # Draw the bounding box on the image
        start_point = (int(bbox[0]), int(bbox[1]))
        end_point = (int(bbox[2]), int(bbox[3]))
        color = (0, 255, 0)  # Green color
        thickness = 2
        image_with_bbox = cv2.rectangle(image, start_point, end_point, color, thickness)

        # Save the image with the bounding box
        cv2.imwrite(output_path, image_with_bbox)

        # print(f"Bounding box drawn on {extracted_frame_filename}")
    # Close the video clip
    video_clip.close()

    question = row["question"]
    answer = row["answer"]
    print("------------")
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print(f"Bounding boxes are surrounding the answer: {answer}")
draw_bounding_boxes(question_id, video_id, path_to_videos, output_dir)
