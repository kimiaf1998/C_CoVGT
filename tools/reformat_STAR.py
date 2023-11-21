"""
This is a script for reformatting the STAR dataset files available online (e.g. STAR_train.json)
into a json file format that only contains the question, answer, start time, end time and 
bounding boxes of the answers.
"""
import pandas as pd
import os.path as osp
import re
import json


##################### TODO modify script variables #####################
# Path of the STAR_orig dataset file you want to convert
INPUT_FILE_PATH = './data/STAR/STAR_train.json'
# File path to write output to

OUTPUT_FILE_PATH = "./C_CoVGT/datasets/star/train_updated_frame_number.json"
########################################################################

OBJECT_CLASS_LOOKUP={}
OBJECT_NAME_LOOKUP={}
OBJECT_REGEX_PATTERN = r'[\w/]+'
VERB_REGEX = r'.+'
CONTACT_REL_REGEX = r'.+'
PERSON_CLASS = "o000"

def load_object_class_lookup():
    file_path = './data/STAR/classes/object_classes.txt'

    # Read the contents of the file and split lines
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Create a dictionary of object string to class
    # e.g. {"person": "o000", "broom": "o001"...}.
    for line in lines:
        (object_class, object_string) = line.split()
        OBJECT_CLASS_LOOKUP[object_string] = object_class
        OBJECT_NAME_LOOKUP[object_class] = object_string

def load_vid_fps_lookup():
    file_path = '../datasets/star/vid_fps_mapping.json'
    # Reading the dictionary from the JSON file
    with open(file_path, 'r') as file:
        return json.load(file)

def calculate_intersection(box1, box2):
    """
    Calculate the intersection of two bounding boxes.

    Parameters:
    - box1, box2: Bounding boxes in the format [x_min, y_min, x_max, y_max].

    Returns:
    - Intersection bounding box in the format [x_min, y_min, x_max, y_max],
      or None if there is no intersection.
    """
    x_min = max(box1[0], box2[0])
    y_min = max(box1[1], box2[1])
    x_max = min(box1[2], box2[2])
    y_max = min(box1[3], box2[3])

    # Check for negative width or height (no intersection)
    if x_max <= x_min or y_max <= y_min:
        return None

    return [x_min, y_min, x_max, y_max]

def calculate_union(box1, box2):
    """
    Calculate the union of two bounding boxes.

    Parameters:
    - box1, box2: Bounding boxes in the format [x_min, y_min, x_max, y_max].

    Returns:
    - Union bounding box in the format [x_min, y_min, x_max, y_max].
    """
    x_min = min(box1[0], box2[0])
    y_min = min(box1[1], box2[1])
    x_max = max(box1[2], box2[2])
    y_max = max(box1[3], box2[3])

    return [x_min, y_min, x_max, y_max]

def get_bbox_of_action(bbox_person, bbox_object):
    intersection = calculate_intersection(bbox_person, bbox_object)
    # if no intersection, use union
    if intersection:
        return intersection
    else:
        return calculate_union(bbox_person, bbox_object)

def get_bboxes_of_action(object_str, situations, question_id, video_id):
    """
    Calculates the bounding boxes of the action at each key frame.

    Parametes:
    - object_str (string): e.g. "sandwich", "book"
    - situations (dict): mapping of key_frame to information in the key_frame (e.g. bounding boxes, object labels, etc)

    Returns:
    dict: mapping of key frames to action bounding box
    """
    object_class = OBJECT_CLASS_LOOKUP[object_str]
    # find the bounding box of this object at each keyframe
    bboxes = {}
    for key_frame in situations:
        bbox_labels = situations[key_frame]["bbox_labels"]
        # can only get the action bounding box if we have the person and object bboxes
        if object_class in bbox_labels and PERSON_CLASS in bbox_labels:
            # get the bounding box coordinates for the object
            index_of_object = bbox_labels.index(object_class)
            bbox_of_object = situations[key_frame]["bbox"][index_of_object]
            # get bounding box of person
            index_of_person = bbox_labels.index(PERSON_CLASS)
            bbox_of_person = situations[key_frame]["bbox"][index_of_person]
            # bbox of the action 
            bbox_of_action = get_bbox_of_action(bbox_of_person, bbox_of_object)
            bboxes[key_frame] = bbox_of_action
        else:
            bboxes[key_frame] = None
            # if object_class not in bbox_labels:
            #     print(f"{question_id}/{video_id}: bbox for {object_str} missing from key_frame {key_frame}.")
            # if PERSON_CLASS not in bbox_labels:
            #     print(f"{question_id}/{video_id}: bbox for person missing from key_frame {key_frame}.")
    return bboxes

def get_bboxes_of_object(object_str, situations, question_id):
    """
    Calculates the bounding boxes of the object at each key frame.

    Parametes:
    - object_str (string): e.g. "sandwich", "book"
    - situations (dict): mapping of key_frame to information in the key_frame (e.g. bounding boxes, object labels, etc)

    Returns:
    dict: mapping of key frames to object bounding box
    """
    object_class = OBJECT_CLASS_LOOKUP[object_str]
    # find the bounding box of this object at each keyframe
    bboxes = {}
    for key_frame in situations:
        # Find the index of the object we are interested in
        if object_class in situations[key_frame]["bbox_labels"]:
            index_of_object = situations[key_frame]["bbox_labels"].index(object_class)
            # get the bounding box coordinates for the object
            bbox = situations[key_frame]["bbox"][index_of_object]
            bboxes[key_frame] = bbox
        else:
            bboxes[key_frame] = None
            # print(f"{question_id}: Bbox for {object_str} missing from key_frame {key_frame}.")
    return bboxes

def get_bboxes_Interaction_T1(row):
    # Regular expression pattern to match "The [obj]."
    pattern = re.compile(fr'The ({OBJECT_REGEX_PATTERN})\.')
    # Extract the [obj] from the string
    object = pattern.match(row["answer"]).group(1)
    return get_bboxes_of_object(object, row["situations"], row["question_id"])

def get_bboxes_Interaction_T2(row):
    """
    Calculates the bounding boxes for the answer to an Interaction_T2 question.
    
    Interaction_T2 questions take the form:

    Question: What did the person do with the [Obj]?
    Answer: [Verb]ed.
    We calculate the bounding box of the verb/action at each key frame. 

    Parameters:
    - row (pandas Series): a row representing a video clip, question and answer

    Returns:
    dict: mapping of key frames to the bounding box
    """
    pattern = re.compile(fr'What did the person do with the ({OBJECT_REGEX_PATTERN})?')
    object = pattern.match(row["question"]).group(1)
    return get_bboxes_of_action(object, row["situations"], row["question_id"], row["video_id"])

def get_bboxes_Interaction_T3(row):
    """
    Calculates the bounding boxes for the answer to an Interaction_T3 question.
    
    Interaction_T3 questions take the form:
        Question: What did the person do while they were [Contact_Rel] the [Obj]?
        Answer: [Verb]ed the [Obj].
    We calculate the bounding box of the verb/action at each key frame. 

    Parameters:
    - row (pandas Series): a row representing a video clip, question and answer

    Returns:
    dict: mapping of key frames to the bounding box
    """
    pattern = re.compile(fr'{VERB_REGEX}? the ({OBJECT_REGEX_PATTERN})\.')
    object = pattern.match(row["answer"]).group(1)
    return get_bboxes_of_action(object, row["situations"], row["question_id"], row["video_id"])

def get_bboxes_Interaction_T4(row):
    """
    Calculates the bounding boxes for the answer to an Interaction_T4 question.
    
    Interaction_T4 questions take the form:
        Question: What did the person do while they were [Contact_Rel1] the [Obj1] and [Contact_Rel2] the [Obj2]?
        Answer: [Verb]ed the [Obj].
    We calculate the bounding box of the verb/action at each key frame. 

    Parameters:
    - row (pandas Series): a row representing a video clip, question and answer

    Returns:
    dict: mapping of key frames to the bounding box
    """
    pattern = re.compile(fr'{VERB_REGEX}? the ({OBJECT_REGEX_PATTERN})\.')
    object = pattern.match(row["answer"]).group(1)
    return get_bboxes_of_action(object, row["situations"], row["question_id"], row["video_id"])

def get_bboxes_Sequence_T1(row):
    """
    Calculates the bounding boxes for the answer to an Sequence_T1 question.
    
    Sequence_T1 questions take the form:
        Question: Which object did the person [Verb1] after they [Verb2]ed the [Obj2]?
        Answer: The [Obj1].
    We calculate the bounding box of Obj1

    Parameters:
    - row (pandas Series): a row representing a video clip, question and answer

    Returns:
    dict: mapping of key frames to the bounding box
    """
    pattern = re.compile(fr'The ({OBJECT_REGEX_PATTERN})\.')
    object_1 = pattern.match(row["answer"]).group(1)
    return get_bboxes_of_object(object_1, row["situations"], row["question_id"])

def get_bboxes_Sequence_T2(row):
    """
    Calculates the bounding boxes for the answer to an Sequence_T2 question.
    
    Sequence_T2 questions take the form:
        Question: Which object did the person [Verb1] before they [Verb2]ed the [Obj2]?
        Answer: The [Obj1].
    We calculate the bounding box of Obj1

    Parameters:
    - row (pandas Series): a row representing a video clip, question and answer

    Returns:
    dict: mapping of key frames to the bounding box
    """
    pattern = re.compile(fr'The ({OBJECT_REGEX_PATTERN})\.')
    object_1 = pattern.match(row["answer"]).group(1)
    return get_bboxes_of_object(object_1, row["situations"], row["question_id"])

def get_bboxes_Sequence_T3(row):
    """
    Calculates the bounding boxes for the answer to an Sequence_T3 question.
    
    Sequence_T3 questions take the form:
        Question: What happened after the person [Verb1]ed the [Obj1]?
        Answer: [Verb2]ed the [Obj2].
    We calculate the bounding box of the action

    Parameters:
    - row (pandas Series): a row representing a video clip, question and answer

    Returns:
    dict: mapping of key frames to the bounding box
    """
    pattern = re.compile(fr'{VERB_REGEX}? the ({OBJECT_REGEX_PATTERN})\.')
    object_2 = pattern.match(row["answer"]).group(1)
    return get_bboxes_of_action(object_2, row["situations"], row["question_id"], row["video_id"])

def get_bboxes_Sequence_T4(row):
    """
    Calculates the bounding boxes for the answer to an Sequence_T4 question.
    
    Sequence_T4 questions take the form:
        Question: What happened before the person [Verb1]ed the [Obj1]?
        Answer: [Verb2]ed the [Obj2].
    We calculate the bounding box of the action

    Parameters:
    - row (pandas Series): a row representing a video clip, question and answer

    Returns:
    dict: mapping of key frames to the bounding box
    """
    pattern = re.compile(fr'{VERB_REGEX}? the ({OBJECT_REGEX_PATTERN})\.')
    object_2 = pattern.match(row["answer"]).group(1)
    return get_bboxes_of_action(object_2, row["situations"], row["question_id"], row["video_id"])

def get_bboxes_Sequence_T5(row):
    """
    Calculates the bounding boxes for the answer to an Sequence_T5 question.
    
    Sequence_T5 questions take the form:
        Question: What did the person do to the [Obj2] after [Verb1]ing the [Obj1]?
        Answer: [Verb2]ed.
    We calculate the bounding box of the action

    Parameters:
    - row (pandas Series): a row representing a video clip, question and answer

    Returns:
    dict: mapping of key frames to the bounding box
    """
    pattern = re.compile(fr'What did the person do to the ({OBJECT_REGEX_PATTERN})? after {VERB_REGEX}? the {OBJECT_REGEX_PATTERN}?\?')
    object_2 = pattern.match(row["question"]).group(1)
    return get_bboxes_of_action(object_2, row["situations"], row["question_id"], row["video_id"])

def get_bboxes_Sequence_T6(row):
    """
    Calculates the bounding boxes for the answer to an Sequence_T6 question.
    
    Sequence_T6 questions take the form:
        Question: What did the person do to the [Obj2] before [Verb1]ing the [Obj1]?
        Answer: [Verb2]ed.
    We calculate the bounding box of the action

    Parameters:
    - row (pandas Series): a row representing a video clip, question and answer

    Returns:
    dict: mapping of key frames to the bounding box
    """
    pattern = re.compile(fr'What did the person do to the ({OBJECT_REGEX_PATTERN})? before {VERB_REGEX}? the {OBJECT_REGEX_PATTERN}?\?')
    object_2 = pattern.match(row["question"]).group(1)
    return get_bboxes_of_action(object_2, row["situations"], row["question_id"], row["video_id"])

def get_bboxes_Prediction_T1(row):
    """
    Calculates the bounding boxes for the answer to an Prediction_T1 question.
    
    Prediction_T1 questions take the form:
        Question: What will the person do next?
        Answer: [Verb] the [Obj].
    We calculate the bounding box of the action

    Parameters:
    - row (pandas Series): a row representing a video clip, question and answer

    Returns:
    dict: mapping of key frames to the bounding box
    """
    pattern = re.compile(fr'{VERB_REGEX}? the ({OBJECT_REGEX_PATTERN})\.')
    object = pattern.match(row["answer"]).group(1)
    return get_bboxes_of_action(object, row["situations"], row["question_id"], row["video_id"])

def get_bboxes_Prediction_T2(row):
    """
    Calculates the bounding boxes for the answer to a Prediction_T2 question.
    
    Prediction_T2 questions take the form:
        Question: What will the person do next with the [Obj]?
        Answer: [Verb].
    We calculate the bounding box of the action

    Parameters:
    - row (pandas Series): a row representing a video clip, question and answer

    Returns:
    dict: mapping of key frames to the bounding box
    """
    pattern = re.compile(fr'What will the person do next with the ({OBJECT_REGEX_PATTERN})\?')
    object = pattern.match(row["question"]).group(1)
    return get_bboxes_of_action(object, row["situations"], row["question_id"], row["video_id"])

def get_bboxes_Prediction_T3(row):
    """
    Calculates the bounding boxes for the answer to a Prediction_T3 question.
    
    Prediction_T3 questions take the form:
        Question: Which object would the person [Verb] next? 
        Answer: The [Obj].
    We calculate the bounding box of the object

    Parameters:
    - row (pandas Series): a row representing a video clip, question and answer

    Returns:
    dict: mapping of key frames to the bounding box
    """
    pattern = re.compile(fr'The ({OBJECT_REGEX_PATTERN})\.')
    object = pattern.match(row["answer"]).group(1)
    return get_bboxes_of_object(object, row["situations"], row["question_id"])

def get_bboxes_Prediction_T4(row):
    """
    Calculates the bounding boxes for the answer to a Prediction_T4 question.
    
    Prediction_T4 questions take the form:
        Question: Which object would the person [Verb2] next after they [Verb1] the [Obj1]?
        Answer: The [Obj2].
    We calculate the bounding box of the object

    Parameters:
    - row (pandas Series): a row representing a video clip, question and answer

    Returns:
    dict: mapping of key frames to the bounding box
    """
    pattern = re.compile(fr'The ({OBJECT_REGEX_PATTERN})\.')
    object = pattern.match(row["answer"]).group(1)
    return get_bboxes_of_object(object, row["situations"], row["question_id"])

def get_bboxes_Feasibility_T1(row):
    """
    Calculates the bounding boxes for the answer to a Feasibility_T1 question.
    
    Feasibility_T1 questions take the form:
        Question: Which other object is possible to be [Verb]ed by the person?
        Answer: The [Obj].
    We calculate the bounding box of the object

    Parameters:
    - row (pandas Series): a row representing a video clip, question and answer

    Returns:
    dict: mapping of key frames to the bounding box
    """
    pattern = re.compile(fr'The ({OBJECT_REGEX_PATTERN})\.')
    object = pattern.match(row["answer"]).group(1)
    return get_bboxes_of_object(object, row["situations"], row["question_id"])

def get_bboxes_Feasibility_T2(row):
    """
    Calculates the bounding boxes for the answer to a Feasibility_T2 question.
    
    Feasibility_T2 questions take the form:
        Question: What else is the person able to do with the [Obj]?
        Answer: [Verb] the [Obj].
    We calculate the bounding box of the action

    Parameters:
    - row (pandas Series): a row representing a video clip, question and answer

    Returns:
    dict: mapping of key frames to the bounding box
    """
    pattern = re.compile(fr'{VERB_REGEX}? the ({OBJECT_REGEX_PATTERN})\.')
    object = pattern.match(row["answer"]).group(1)
    return get_bboxes_of_action(object, row["situations"], row["question_id"], row["video_id"])

def get_bboxes_Feasibility_T3(row):
    """
    Calculates the bounding boxes for the answer to a Feasibility_T3 question.
    
    Feasibility_T3 questions take the form:
        Question: Which object is possible to be [Verb1]ed when the person is [Spatial_Rel] the [Obj2]?
        Answer: The [Obj1].
    We calculate the bounding box of the object

    Parameters:
    - row (pandas Series): a row representing a video clip, question and answer

    Returns:
    dict: mapping of key frames to the bounding box
    """
    pattern = re.compile(fr'The ({OBJECT_REGEX_PATTERN})\.')
    object = pattern.match(row["answer"]).group(1)
    return get_bboxes_of_object(object, row["situations"], row["question_id"])

def get_bboxes_Feasibility_T4(row):
    """
    Calculates the bounding boxes for the answer to a Feasibility_T4 question.
    
    Feasibility_T4 questions take the form:
        Question: What is the person able to do when they are [Spatial_Rel] the [Obj2]?
        Answer: [Verb1] the [Obj1].
    We calculate the bounding box of the action

    Parameters:
    - row (pandas Series): a row representing a video clip, question and answer

    Returns:
    dict: mapping of key frames to the bounding box
    """
    pattern = re.compile(fr'{VERB_REGEX}? the ({OBJECT_REGEX_PATTERN})\.')
    object = pattern.match(row["answer"]).group(1)
    return get_bboxes_of_action(object, row["situations"], row["question_id"], row["video_id"])

def get_bboxes_Feasibility_T5(row):
    """
    Calculates the bounding boxes for the answer to a Feasibility_T5 question.
    
    Feasibility_T5 questions take the form:
        Question: Which object is the person able to [Verb1] after [Verb2]ing the [Obj2]?
        Answer: The [Obj1].
    We calculate the bounding box of the object

    Parameters:
    - row (pandas Series): a row representing a video clip, question and answer

    Returns:
    dict: mapping of key frames to the bounding box
    """
    pattern = re.compile(fr'The ({OBJECT_REGEX_PATTERN})\.')
    object = pattern.match(row["answer"]).group(1)
    return get_bboxes_of_object(object, row["situations"], row["question_id"])

def get_bboxes_Feasibility_T6(row):
    """
    Calculates the bounding boxes for the answer to a Feasibility_T6 question.
    
    Feasibility_T6 questions take the form:
        Question: What is the person able to do after [Verb1]ing the [Obj1]?
        Answer: [Verb2] the [Obj2].
    We calculate the bounding box of the action

    Parameters:
    - row (pandas Series): a row representing a video clip, question and answer

    Returns:
    dict: mapping of key frames to the bounding box
    """
    pattern = re.compile(fr'{VERB_REGEX}? the ({OBJECT_REGEX_PATTERN})\.')
    object = pattern.match(row["answer"]).group(1)
    return get_bboxes_of_action(object, row["situations"], row["question_id"], row["video_id"])

def update_frame_numbers_using_new_framerate(row, new_fps=3):
    """
    Map a frame number using the original framerate to the corresponding frame number
    when using a fixed framerate (defaults to 3 fps).

    For example, if a video framerate was orginally 24.5 fps, frame 49 would correspond
    to the 2 second mark in the video (49 frames/24.5 fps = 2 seconds).
    The corresponding frame for the video with framerate 3 fps would be 
    3 fps * 2 seconds = frame 6.
    Therefore, frame 49 with the original framerate corresponds to frame 6 with a 
    framerate of 3 fps.
    
    For any video and frame, let k = keyframe number (e.g. 49) and let 
    f = original fps of the video (e.g. 24.5).
    The corresponding frame number at 3 fps = k / f * 3
    """
    new_frame_number_to_bboxes = {}
    for frame_number, bbox in row["bboxes"].items():
        if bbox == None:
            continue
        new_frame_number = int(frame_number) / VID_FPS_LOOKUP[row["video_id"]] * new_fps
        # fractions of a frame (e.g. 26.55) don't exist so round to 
        # nearest whole frame number (e.g. 27)
        rounded_frame_number = round(new_frame_number)
        formatted_frame_number = str(rounded_frame_number).zfill(6)
        # add to dict
        if formatted_frame_number in new_frame_number_to_bboxes:
            new_frame_number_to_bboxes[formatted_frame_number].append(bbox)
        else:
            new_frame_number_to_bboxes[formatted_frame_number] = [bbox]
    
    # When multiple frames from the original framerate are mapped to the same frame 
    # using 3fps (e.g. frame 245, 244 at 30fps are both mapped to frame 24 at 3fps),
    # take the average of the bounding boxes
    new_frame_number_to_avg_bbox = {}
    for frame_number, bboxes in new_frame_number_to_bboxes.items():
        num_bboxes = len(bboxes)
        avg_bbox = [0, 0, 0, 0]
        for bbox in bboxes:
            avg_bbox[0] += bbox[0]
            avg_bbox[1] += bbox[1]
            avg_bbox[2] += bbox[2]
            avg_bbox[3] += bbox[3]
        avg_bbox = [value / num_bboxes for value in avg_bbox]
        new_frame_number_to_avg_bbox[frame_number] = avg_bbox
        
    # replace with new frame numbers    
    row["bboxes"] = new_frame_number_to_avg_bbox

def main():
    results = []

    # load mapping e.g. {"person": "o000", "broom": "o001"...}.
    load_object_class_lookup()

    # Read the JSON file into a DataFrame
    df = pd.read_json(INPUT_FILE_PATH)

    for (index, row) in df.iterrows():
        result = {}
        result["question_id"] = row["question_id"]
        result["video_id"] = row["video_id"]
        result["start"] = row["start"]
        result["end"] = row["end"]
        result["question"] = row["question"]
        result["answer"] = row["answer"]
        result["choices"] = [{'choice_id': choice['choice_id'], 'choice': choice['choice']} for choice in row["choices"]]

        # get the bounding box of the answer
        if row["question_id"].startswith("Interaction_T1"):
            result["bboxes"] = get_bboxes_Interaction_T1(row)
        elif row["question_id"].startswith("Interaction_T2"):
            result["bboxes"] = get_bboxes_Interaction_T2(row)
        elif row["question_id"].startswith("Interaction_T3"):
            result["bboxes"] = get_bboxes_Interaction_T3(row)
        elif row["question_id"].startswith("Interaction_T4"):
            result["bboxes"] = get_bboxes_Interaction_T4(row)
        elif row["question_id"].startswith("Sequence_T1"):
            result["bboxes"] = get_bboxes_Sequence_T1(row)
        elif row["question_id"].startswith("Sequence_T2"):
            result["bboxes"] = get_bboxes_Sequence_T2(row)
        elif row["question_id"].startswith("Sequence_T3"):
            result["bboxes"] = get_bboxes_Sequence_T3(row)
        elif row["question_id"].startswith("Sequence_T4"):
            result["bboxes"] = get_bboxes_Sequence_T4(row)
        elif row["question_id"].startswith("Sequence_T5"):
            result["bboxes"] = get_bboxes_Sequence_T5(row)
        elif row["question_id"].startswith("Sequence_T6"):
            result["bboxes"] = get_bboxes_Sequence_T6(row)
        elif row["question_id"].startswith("Prediction_T1"):
            result["bboxes"] = get_bboxes_Prediction_T1(row)
        elif row["question_id"].startswith("Prediction_T2"):
            result["bboxes"] = get_bboxes_Prediction_T2(row)
        elif row["question_id"].startswith("Prediction_T3"):
            result["bboxes"] = get_bboxes_Prediction_T3(row)
        elif row["question_id"].startswith("Prediction_T4"):
            result["bboxes"] = get_bboxes_Prediction_T4(row)
        elif row["question_id"].startswith("Feasibility_T1"):
            result["bboxes"] = get_bboxes_Feasibility_T1(row)
        elif row["question_id"].startswith("Feasibility_T2"):
            result["bboxes"] = get_bboxes_Feasibility_T2(row)
        elif row["question_id"].startswith("Feasibility_T3"):
            result["bboxes"] = get_bboxes_Feasibility_T3(row)
        elif row["question_id"].startswith("Feasibility_T4"):
            result["bboxes"] = get_bboxes_Feasibility_T4(row)
        elif row["question_id"].startswith("Feasibility_T5"):
            result["bboxes"] = get_bboxes_Feasibility_T5(row)
        elif row["question_id"].startswith("Feasibility_T6"):
            result["bboxes"] = get_bboxes_Feasibility_T6(row)
        results.append(result)

        update_frame_numbers_using_new_framerate(result, new_fps=3)

    # Write the results to a JSON file
    with open(OUTPUT_FILE_PATH, 'w') as json_file:
        json.dump(results, json_file, indent=2)  # indent parameter for pretty formatting

# load mapping of (video id -> fps) e.g. {'R3O7U': 29.97003, ..}
VID_FPS_LOOKUP=load_vid_fps_lookup()
main()

# Notes:
# - There are no "Feasibility_T1" questions in the train dataset
# - the object's bounding box is missing from some key frames
# - bounding boxes are x_min, y_min, x_max, y_max (inferred from this: https://github.com/csbobby/STAR_Benchmark/blob/main/code/visualization_tools/qa_visualization.py#L70C62-L70C62)
