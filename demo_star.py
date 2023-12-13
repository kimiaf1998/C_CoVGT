import argparse

import torch
import os
import os.path as osp
import numpy as np
from args import get_args, get_parser
from PIL import Image
# import matplotlib.pyplot as plt

from transformers import RobertaTokenizerFast

from models.Tube_CoVGT import build_model
from tools.object_align import align
from tools.postprocess import PostProcess
from tools.util import tokenize, load_file, transform_bb, load_model_by_key
from util import get_mask


def get_video_info(video_name, qid):
    video_path = f'/data/kimia/hdd3_mount/kimia/data/STAR/frames_orig_fps'
    vid_clips = load_file(os.path.join(args.dataset_dir, args.dataset) + f'/clips_val.json')[qid]

    video_root_dir = '/data/kimia/hdd2_mount/kimia_data/projects/data/STAR'
    video_feature_path = f'{video_root_dir}/pre_features'
    app_feats = []
    roi_feats, roi_bboxs = [], []
    video_frame_ids = []
    fid = vid_clips[0][0]
    # features indices starts from 0 while frames 1
    img = Image.open(f'{video_path}/{video_id}/{int(fid) + 1:06}.png')
    width, height = img.size
    img.close()

    for cid, clip in enumerate(vid_clips):
        clip_feat, clip_rfeat, clip_rbbox = [], [], []
        clip_frame_ids = []
        for fid in clip:
            clip_frame_ids.append(fid)
            fid = int(fid)
            frame_feat_file = osp.join(video_feature_path, f'frame_feat/{video_name}/{fid:06d}.npy')
            frame_feat = np.load(frame_feat_file)
            clip_feat.append(frame_feat)

            region_feat_file = osp.join(video_feature_path, f'bbox/{video_name}/{fid:06d}.npz')
            region_feat = np.load(region_feat_file)
            clip_rfeat.append(region_feat['x'])
            clip_rbbox.append(region_feat['bbox'])
        app_feats.append(clip_feat)
        feats = np.asarray(clip_rfeat)
        bboxes = np.asarray(clip_rbbox)
        video_frame_ids.append(clip_frame_ids)
        vid_feat_aln, vid_bbox_aln = align(feats, bboxes, video_name, cid)
        roi_feats.append(vid_feat_aln)
        roi_bboxs.append(vid_bbox_aln)
    frame_feat = np.asarray(app_feats)
    app_feats = torch.from_numpy(frame_feat).type(torch.float32)

    roi_feats = np.asarray(roi_feats)
    roi_feats = torch.from_numpy(roi_feats).type(torch.float32)

    roi_bboxs = np.asarray(roi_bboxs)
    bbox_feats = transform_bb(roi_bboxs, width, height)  # [x1,y1,x2,y2,area]
    bbox_feats = torch.from_numpy(bbox_feats).type(torch.float32)

    region_feats = torch.cat((roi_feats, bbox_feats), dim=-1)

    # print(region_feats.shape, app_feats.shape)
    # print(bbox_feats.shape)

    # return bbox_feats without area
    return region_feats, app_feats, video_frame_ids, (width, height)


if __name__ == "__main__":
    args = get_args()
    device = args.device

    # load models
    model, _ = build_model(args)
    model.to(device)
    print("models loaded")

    postprocessors = PostProcess()

    # load checkpoint
    assert args.load
    model.load_state_dict(load_model_by_key(model, args.load))
    print("checkpoint loaded")

    # load video (with eventual start & end) & caption demo examples
    question_txt = args.question
    qid = args.qid
    answer_txt = args.answer
    answer_choices = args.choices
    video_id = args.vid_id

    # load pre-extracted features
    video_o, video_f, video_frame_ids, vid_orig_size = get_video_info(video_id, qid)

    max_object_len = video_o.size(1)

    ################################################################

    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    answer_txts = [question_txt + f' {tokenizer.sep_token} ' + opt for opt in answer_choices]
    ans_id = answer_choices.index(answer_txt)

    # fetch tokens of questions & answer
    ans_token_ids, _ = tokenize(
        answer_txts,
        tokenizer,
        add_special_tokens=True,
        max_length=args.amax_words,
        dynamic_padding=False,
        truncation=True
    )
    seq_len = torch.tensor([len(ans) for ans in ans_token_ids], dtype=torch.long).unsqueeze(0)
    ans_token_ids = ans_token_ids.unsqueeze(0).to(device)
    # inputs needs to be batch-wised
    video_f = video_f.unsqueeze(0).to(device)
    video_o = video_o.unsqueeze(0).to(device)
    samples = (video_o, video_f)
    questions = torch.tensor([0], dtype=torch.long).unsqueeze(0).to(device)

    answer_mask = (ans_token_ids != tokenizer.pad_token_id).float().to(device)  # RobBERETa
    video_mask = get_mask(torch.tensor([video_o.size(1)]).to(device), video_o.size(1))

    model.eval()
    with torch.no_grad():  # forward through the models
        fusion_proj, answer_proj, tube_pred = model(
            samples,
            question=questions,
            answer=ans_token_ids,
            text_mask=answer_mask,
            video_mask=video_mask,
            seq_len=seq_len,
            localization=True,
        )

        # fusion_proj = fusion_proj.unsqueeze(2)
        # predicts = torch.bmm(answer_proj, fusion_proj).squeeze()
        #
        # predicted = torch.max(predicts, dim=1).indices.cpu()
        #
        # # calculate textual answer accuracy
        #
        # output = {}
        # pred_id = predicted
        # pred = answer_choices[pred_id]
        # output.update({'question': question_txt, 'prediction': pred, 'answer': answer_txt})
        #
        # # convert predicts from relative [0, 1] to absolute [0, height] coordinates
        # results = PostProcess()(tube_pred, vid_orig_size).to(device)
        #
        # bbox_res = {}  # maps image_id to the coordinates of the detected box
        # video_frame_ids = np.asarray(video_frame_ids).reshape(-1)
        # print("result shape:", len(results))
        # print("result shape:", video_frame_ids.shape)
        # for frm_id, result in zip(video_frame_ids, results):
        #     bbox_res[frm_id] = result.detach().cpu().tolist()
        # output.update(bbox_res)
        #
        # # # create output dirs
        # # if not os.path.exists(args.output_dir):
        # #     os.makedirs(args.output_dir)
        # # if not os.path.exists(os.path.join(args.output_dir, vid_path.split("/")[-1][:-4])):
        # #     os.makedirs(os.path.join(args.output_dir, vid_path.split("/")[-1][:-4]))
        # # # extract actual images from the video to process them adding boxes
        # # os.system(
        # #     f'ffmpeg -i {vid_path} -ss {ss} -t {t} -qscale:v 2 -r {extracted_fps} {os.path.join(args.output_dir, vid_path.split("/")[-1][:-4], "%05d.jpg")}'
        # # )
        # # for img_id in image_ids[0]:
        # #     # load extracted image
        # #     img_path = os.path.join(
        # #         args.output_dir,
        # #         vid_path.split("/")[-1][:-4],
        # #         str(int(img_id) + 1).zfill(5) + ".jpg",
        # #     )
        # #     img = Image.open(img_path).convert("RGB")
        # #     imgw, imgh = img.size
        # #     fig, ax = plt.subplots()
        # #     ax.axis("off")
        # #     ax.imshow(img, aspect="auto")
        # #
        # #     if (
        # #             pred_steds[0] <= img_id < pred_steds[1]
        # #     ):  # add predicted box if the image_id is in the predicted start and end
        # #         x1, y1, x2, y2 = vidstg_res[img_id]["boxes"][0]
        # #         w = x2 - x1
        # #         h = y2 - y1
        # #         rect = plt.Rectangle(
        # #             (x1, y1), w, h, linewidth=2, edgecolor="#FAFF00", fill=False
        # #         )
        # #         ax.add_patch(rect)
        # #
        # #     fig.set_dpi(100)
        # #     fig.set_size_inches(imgw / 100, imgh / 100)
        # #     fig.tight_layout(pad=0)
        # #
        # #     # save image with eventual box
        # #     fig.savefig(
        # #         img_path,
        # #         format="jpg",
        # #     )
        # #     plt.close(fig)
        # #
        # # # save video with tube
        # # os.system(
        # #     f"ffmpeg -r {extracted_fps} -pattern_type glob -i '{os.path.join(args.output_dir, vid_path.split('/')[-1][:-4])}/*.jpg' -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' -r {extracted_fps} -crf 25 -c:v libx264 -pix_fmt yuv420p -movflags +faststart {os.path.join(args.output_dir, vid_path.split('/')[-1])}"
        # # )
