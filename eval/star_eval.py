from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import tools.dist as dist

import json
from functools import reduce
from tools.box_ops import box_iou, box_cxcywh_to_xyxy


class STARiouEvaluator:
    def __init__(
        self,
        targets: list,
        iou_thresholds: list = [0.3, 0.5],
    ):
        """
        :param targets: batch of videos
        :param iou_thresholds: IoU thresholds for the vIoU metrics
        """

        self.iou_thresholds = iou_thresholds
        self.targets = targets

        self.video_ids = targets["video_id"]
        self.question_ids = targets["question_id"]
        self.gt_bboxes = targets["bboxes"]
        self.gt_bboxes_mask = targets["bboxes_mask"]
        self.frame_mapping = targets["frame_mapping"]

    def evaluate(self, predictions: List[List]): # predictions => BxTxnum_quesriesx4
        print("predictions:", predictions.shape)
        if len(predictions) < len(self.targets):
            raise RuntimeError(
                f"{len(self.targets) - len(predictions)} box predictions missing"
            )
        vid_metrics = {}

        for idx, pred_bboxes in enumerate(predictions): # iterate on video batches
            pred_bboxes= box_cxcywh_to_xyxy(pred_bboxes)
            gt_bboxes = self.gt_bboxes[idx]
            gt_bboxes = gt_bboxes.to(pred_bboxes.device)
            gt_bboxes_mask = self.gt_bboxes_mask[idx]
            video_id = self.video_ids[idx]
            question_id = self.question_ids[idx]

            total_annotated_frames = torch.sum(gt_bboxes_mask).item()

            # no bbox annotated for this video
            if total_annotated_frames <= 0:
                print("no bbox annotated for video:", video_id)
                continue

            viou = 0
            vid_metrics[question_id] = {}

            for pred_bb, gt_bb, gt_bb_mask in zip(pred_bboxes, gt_bboxes, gt_bboxes_mask):  # iterate on all frames of the annotated moment to update GT metrics
                if gt_bb_mask: # if frame annotated
                    iou_matrix, _ = box_iou(gt_bb, pred_bb)
                    max_preds, _ = torch.max(iou_matrix, dim=1)
                    iou = torch.mean(max_preds).item()
                    viou += iou

            # compute viou@R
            viou = viou / total_annotated_frames
            vid_metrics[question_id].update({"viou" : viou})
            recalls = {thresh: 0 for thresh in self.iou_thresholds}
            for thresh in self.iou_thresholds:
                if viou > thresh:
                    recalls[thresh] += 1
            vid_metrics[question_id].update(
                {
                    f"viou@{thresh}": recalls[thresh]
                    for thresh in self.iou_thresholds
                }
            )

        return vid_metrics


class STAREvaluator(object):
    def __init__(
        self,
        targets: list,
        iou_thresholds=[0.3, 0.5],
        save_pred=False,
    ):
        """
        :param targets: batch of videos
        :param iou_thresholds: IoU thresholds for the vIoU metrics
        :param save_pred: whether to save predictions in the output of summarize
        """
        self.evaluator = STARiouEvaluator(
            targets,
            iou_thresholds=iou_thresholds,
        )
        self.predictions = {}
        self.results = None
        self.iou_thresholds = iou_thresholds
        self.save_pred = save_pred
        self.pred_sted = {}

    def update(self, predictions):
        self.predictions = predictions

    def summarize(self):
        self.results = self.evaluator.evaluate(
            self.predictions
        )
        categories = set('_'.join(x.split("_")[1:3]) for x in self.results.keys())
        print("self.results:", self.results)
        print("categories:", categories)
        metrics = {}
        counter = {}
        for category in categories:  # init metrics
            metrics.update({category: {"viou": 0}})
            for thresh in self.iou_thresholds:
                metrics[category].update({f"viou@{thresh}" : 0})
            counter.update({category : 0})
        for question_id, x in self.results.items():  # sum results
            print("metrics:", metrics)
            question_cat = '_'.join(question_id.split("_")[1:3])
            metrics[question_cat]["viou"] += x["viou"]
            for thresh in self.iou_thresholds:
                metrics[question_cat][f"viou@{thresh}"] += x[f"viou@{thresh}"]
            counter[question_cat] += 1
        for category in categories:  # average results per category
            for key in metrics[category]: # used to be qid
                metrics[category][key] = metrics[category][key] / counter[category]
                print(f"{category} {key}: {metrics[category][key]:.4f}")
        out = {
            f"{question_id}_{name}": metrics[question_id][name]
            for question_id in metrics
            for name in metrics[question_id]
        }
        if self.save_pred:
            out["predictions"] = self.predictions
            out["vid_metrics"] = self.results
        return out
