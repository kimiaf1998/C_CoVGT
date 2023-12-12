import torch
import os
import numpy as np
import ffmpeg
import argparse
from args import get_args
from PIL import Image
import matplotlib.pyplot as plt

from transformers import RobertaTokenizerFast

from models.Tube_CoVGT import build_model
from datasets.video_transforms import prepare, make_video_transforms
from tools.postprocess import PostProcess
from tools.util import tokenize

parser = argparse.ArgumentParser(
    "Demo test script", parents=[get_args()]
)
parser.add_argument('--load', default="", type=str, help='Provide a path to the model checkpoint')
parser.add_argument('--vid_id', default="", type=str, help='Provide a video sample id')
parser.add_argument('--question', default="", type=str, help='Provide a question related to the video sample')
parser.add_argument('--answer', default="", type=str, help='Provide an answer to the provided question')
parser.add_argument('--choices', nargs=4, type=int, help='Provide 4 answer choices as a list')

args = parser.parse_args()
device = args.device

# load models
model = build_model(args)
model.to(device)
print("models loaded")

postprocessors = PostProcess()

# load checkpoint
assert args.load
checkpoint = torch.load(args.load, map_location="cpu")

model.load_state_dict(checkpoint["models"], strict=False)
print("checkpoint loaded")

# load video (with eventual start & end) & caption demo examples
question_txt = args.question
answer_txt = args.answer
answer_choices = args.choices
video_id = args.video_id

# load pre-extracted features
video_o, _, video_f = get_video_feat_star(vid_id, qid, width, height)
video = (video_o, video_f)

print("video_o shape:", video_o.shape)  # Txnum_queriesxF
max_object_len = video_o.size(1)

################################################################


tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
answer_txts = [question_txt + f' {tokenizer.sep_token} ' + opt for opt in answer_choices]
ans_id = answer_choices.index(answer_txt)
num_words = len(answer_txts)

# fetch tokens of questions & answer
ans_token_ids, _ = tokenize(
    answer_txts,
    tokenizer,
    add_special_tokens=True,
    max_length=num_words,
    dynamic_padding=False,
    truncation=True
)
seq_len = torch.tensor([len(ans) for ans in ans_token_ids], dtype=torch.long)

bs, numc, numf, max_object_num, _ = video_o.size()

with torch.no_grad():  # forward through the models
    fusion_proj, answer_proj, tube_pred = model(
        video,
        answer=ans_token_ids,
        seq_len=seq_len,
        localization=True,
    )

    fusion_proj = fusion_proj.unsqueeze(2)
    predicts = torch.bmm(answer_proj, fusion_proj).squeeze()

    predicted = torch.max(predicts, dim=1).indices.cpu()

    # calculate textual answer accuracy

    output = {}
    pred_id = predicted
    pred = answer_choices[pred_id]
    output.update({'question': question_txt, 'prediction': pred, 'answer': answer_txt})

    # convert predicts from relative [0, 1] to absolute [0, height] coordinates
    results = PostProcess()(tube_pred["pred_boxes"], vid_orig_size)


################################################################

image_ids = [[k for k in range(len(images_list))]]

# video transforms
empty_anns = []  # empty targets as placeholders for the transforms
placeholder_target = prepare(w, h, empty_anns)
placeholder_targets_list = [placeholder_target] * len(images_list)
transforms = make_video_transforms("test", cautious=True, resolution=args.resolution)
images, targets = transforms(images_list, placeholder_targets_list)
samples = NestedTensor.from_tensor_list([images], False)
if args.stride:
    samples_fast = samples.to(device)
    samples = NestedTensor.from_tensor_list([images[:, :: args.stride]], False).to(
        device
    )
else:
    samples_fast = None
durations = [len(targets)]

with torch.no_grad():  # forward through the models
    # encoder
    memory_cache = model(
        samples,
        durations,
        captions,
        encode_and_save=True,
        samples_fast=samples_fast,
    )
    # decoder
    outputs = model(
        samples,
        durations,
        captions,
        encode_and_save=False,
        memory_cache=memory_cache,
    )

    pred_steds = postprocessors["vidstg"](outputs, image_ids, video_ids=[0])[
        0
    ]  # (start, end) in terms of image_ids
    orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).to(device)
    results = postprocessors["bbox"](outputs, orig_target_sizes)
    vidstg_res = {}  # maps image_id to the coordinates of the detected box
    for im_id, result in zip(image_ids[0], results):
        vidstg_res[im_id] = {"boxes": [result["boxes"].detach().cpu().tolist()]}

    # create output dirs
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(os.path.join(args.output_dir, vid_path.split("/")[-1][:-4])):
        os.makedirs(os.path.join(args.output_dir, vid_path.split("/")[-1][:-4]))
    # extract actual images from the video to process them adding boxes
    os.system(
        f'ffmpeg -i {vid_path} -ss {ss} -t {t} -qscale:v 2 -r {extracted_fps} {os.path.join(args.output_dir, vid_path.split("/")[-1][:-4], "%05d.jpg")}'
    )
    for img_id in image_ids[0]:
        # load extracted image
        img_path = os.path.join(
            args.output_dir,
            vid_path.split("/")[-1][:-4],
            str(int(img_id) + 1).zfill(5) + ".jpg",
        )
        img = Image.open(img_path).convert("RGB")
        imgw, imgh = img.size
        fig, ax = plt.subplots()
        ax.axis("off")
        ax.imshow(img, aspect="auto")

        if (
                pred_steds[0] <= img_id < pred_steds[1]
        ):  # add predicted box if the image_id is in the predicted start and end
            x1, y1, x2, y2 = vidstg_res[img_id]["boxes"][0]
            w = x2 - x1
            h = y2 - y1
            rect = plt.Rectangle(
                (x1, y1), w, h, linewidth=2, edgecolor="#FAFF00", fill=False
            )
            ax.add_patch(rect)

        fig.set_dpi(100)
        fig.set_size_inches(imgw / 100, imgh / 100)
        fig.tight_layout(pad=0)

        # save image with eventual box
        fig.savefig(
            img_path,
            format="jpg",
        )
        plt.close(fig)

    # save video with tube
    os.system(
        f"ffmpeg -r {extracted_fps} -pattern_type glob -i '{os.path.join(args.output_dir, vid_path.split('/')[-1][:-4])}/*.jpg' -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' -r {extracted_fps} -crf 25 -c:v libx264 -pix_fmt yuv420p -movflags +faststart {os.path.join(args.output_dir, vid_path.split('/')[-1])}"
    )
