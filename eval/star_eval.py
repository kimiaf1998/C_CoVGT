from pathlib import Path
from typing import Dict, List

import numpy as np
import tools.dist as dist

import json
from functools import reduce
from tools.box_ops import np_box_iou


class STARiouEvaluator:
    def __init__(
        self,
        targets: list,
        frame_mapping: dict,
        iou_thresholds: list = [0.3, 0.5],
    ):
        """
        :param targets: batch of videos
        :param frame_mapping: frame idx to real frame number
        :param iou_thresholds: IoU thresholds for the vIoU metrics
        """

        self.iou_thresholds = iou_thresholds
        self.targets = targets
        self.frame_mapping = frame_mapping

    def evaluate(self, predictions: List[Dict]): # predictions => BxTxnum_quesriesx4
        if len(predictions) < len(self.targets):
            raise RuntimeError(
                f"{len(self.img2box) - len(predictions)} box predictions missing"
            )
        vid_metrics = {}
        for preds, target in zip(predictions, self.targets):

            video_id = target["video_id"]
            question_id = target["question_id"]
            gt_bboxes = target["bboxes"]
            pred_bboxes = preds["bboxes"]

            vid_metrics[question_id] = {
            }

            viou = 0

            for pred_bb, gt_bb in zip(pred_bboxes, gt_bboxes):  # iterate on all frames of the annotated moment to update GT metrics
                iou_matrix = np_box_iou(np.array(gt_bb), np.array(pred_bb))
                max_preds, _ = torch.max(iou_matrix, dim=1)
                iou = torch.mean(max_preds)
                viou += iou

            # compute viou@R
            viou = viou / max(len(pred_bboxes), 1)
            vid_metrics[video_id]["viou"] = viou
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
        frame_mapping: dict,
        iou_thresholds=[0.3, 0.5],
        save_pred=False,
    ):
        """
        :param targets: batch of videos
        :param frame_mapping: frame idx to real frame number
        :param iou_thresholds: IoU thresholds for the vIoU metrics
        :param save_pred: whether to save predictions in the output of summarize
        """
        self.evaluator = STARiouEvaluator(
            targets,
            frame_mapping,
            iou_thresholds=iou_thresholds,
        )
        self.predictions = {}
        self.video_predictions = {}
        self.results = None
        self.iou_thresholds = iou_thresholds
        self.save_pred = save_pred
        self.pred_sted = {}

    def update(self, predictions):
        self.predictions.update(predictions)

    def summarize(self):
        self.results = self.evaluator.evaluate(
            self.predictions
        )
        categories = set(x["question_id"].split("_")[:-1] for x in self.results.keys())
        print("categories:", categories)
        metrics = {}
        counter = {}
        for category in categories:  # init metrics
            if self.tmp_loc:
                metrics[category].update({"viou": 0})
            for thresh in self.iou_thresholds:
                metrics[category][f"viou@{thresh}"] = 0
            counter[category] = 0
        for x in self.results.values():  # sum results
            question_id = x["question_id"].split("_")[:-1]
            metrics[question_id]["viou"] += x["viou"]
            for thresh in self.iou_thresholds:
                metrics[question_id][f"viou@{thresh}"] += x[f"viou@{thresh}"]
            counter[question_id] += 1
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
            out["video_predictions"] = self.video_predictions
            out["vid_metrics"] = self.results
        return out
