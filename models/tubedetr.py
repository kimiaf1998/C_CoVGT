# Adapted from
# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
TubeDETR models and criterion classes.
"""
from typing import Dict, Optional

import torch
import torch.distributed
import torch.nn.functional as F
from torch import nn
import math

from tools import box_ops, dist
from torchvision.ops import generalized_box_iou
from tools import box_ops

from .space_time_decoder import build_transformer


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.dropout = dropout
        if dropout:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
            if self.dropout and i < self.num_layers:
                x = self.dropout(x)
        return x


class TubeDecoder(nn.Module):
    """This is the TubeDETR module that performs spatio-temporal video grounding"""

    def __init__(
        self,
        transformer,
        num_queries,
        video_max_len=8*4,
        guided_attn=False,
        sted=False,
    ):
        """
        :param transformer: transformer models
        :param num_queries: number of object queries per frame
        :param video_max_len: maximum number of frames in the models
        :param guided_attn: whether to use guided attention loss
        :param sted: whether to predict start and end proba
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3) # bool (is the ans or not)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        self.video_max_len = video_max_len
        self.guided_attn = guided_attn
        self.sted = sted
        if sted:
            self.sted_embed = MLP(hidden_dim, hidden_dim, 2, 2, dropout=0.5)

    def forward(
        self,
        object_encoding, # (bs, t, numr, dmodel) # TODO to be removed later
        vt_encoding, # (bs, numc, dmodel)
        object_mask, # (numr, numr)
        vt_mask, # (bs, numc)
    ):
        """The forward expects a NestedTensor, which consists of:
           - samples.tensor: batched frames, of shape [n_frames x 3 x H x W]
           - samples.mask: a binary mask of shape [n_frames x H x W], containing 1 on padded pixels

        It returns a dict with the following elements:
           - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, height, width). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
           - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        assert vt_encoding is not None
        # space-time decode

        bs, t, num_queries, _ = object_encoding.size()
        _, numc, _ = vt_encoding.size()
        numf = t//numc
        # query_encoding = object_encoding.view(num_queries, bs*t, -1) # (num_queries, bs*t, dmodel)

        vt_encoding = vt_encoding.flatten(0,1).unsqueeze(0).repeat(1, numf, 1) # (bs, numc, dmodel) -> (bs*numc, dmodel)
                                                                            # -> (1, bs*numc, dmodel)-> (1, bs*t, dmodel)
        vt_mask.reshape(bs * numc, -1).repeat(numf, 1) # (bs, numc) -> (bs*numc, 1) -> (bs*t,1)

        query_embed = self.query_embed.weight

        query_embed = query_embed.unsqueeze(1).repeat(
            1, bs*t, 1
        )  # n_queriesx(BT)xF

        hs = self.transformer(
            # query_encoding=query_encoding,  # (num_queries)x(BT)xF
            query_encoding=query_embed,  # (num_queries)x(BT)xF
            vt_encoding=vt_encoding,  # (1)x(BT)xF
            query_mask=object_mask,  # num_queriesxnum_queries)
            vt_mask=vt_mask,  # (BT)x1
        )
        if self.guided_attn:
            hs, weights, cross_weights = hs
        out = {}

        # outputs heads
        if self.sted:
            outputs_sted = self.sted_embed(hs)

        outputs_coord = self.bbox_embed(hs).sigmoid() # n_layersx(b*t)xnum_queriesx4
        out.update({"pred_boxes": outputs_coord[-1]}) # fetch last-layer output ->  (b*t)xnum_queriesx4
        if self.sted:
            out.update({"pred_sted": outputs_sted[-1]})
        if self.guided_attn:
            out["weights"] = weights[-1]
            out["ca_weights"] = cross_weights[-1]

        return out


class SetCriterion(nn.Module):
    """This class computes the loss for TubeDETR."""

    def __init__(self, losses, sigma=1):
        """Create the criterion.
        Parameters:
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            sigma: standard deviation for the Gaussian targets in the start and end Kullback Leibler divergence loss
        """
        super().__init__()
        self.losses = losses
        self.sigma = sigma

    def loss_boxes(self, outputs, targets):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert "pred_boxes" in outputs

        src_boxes = outputs["pred_boxes"]
        target_boxes = targets["bboxes"] # bxtx1x4
        target_boxes = target_boxes.to(src_boxes.device)
        target_boxes_mask = targets["bboxes_mask"] # bxt

        # keep gt boxes within the annotated moment
        keep_indices = torch.nonzero(target_boxes_mask)
        target_boxes = target_boxes[keep_indices[:,0], keep_indices[:,1], :, :]
        src_boxes = src_boxes[keep_indices[:,0], keep_indices[:,1], :, :]

        src_boxes = src_boxes.reshape(-1, 4) # (b*t)xnum_queriesx4 => (bs*t*num_queries)x4
        src_boxes = box_ops.box_cxcywh_to_xyxy(src_boxes)
        target_boxes = target_boxes.reshape(-1, 4) # (b*t*1)x4

        losses = {}

        giou_matrix = generalized_box_iou(
            target_boxes,
            src_boxes,
        )

        # match predicted bboxes
        matched_bboxes_indices = torch.argmax(giou_matrix, dim=1)
        matched_bboxes = src_boxes[matched_bboxes_indices, ...]
        matched_gious = giou_matrix[..., matched_bboxes_indices]

        losses["loss_giou"] = 1 - torch.mean(matched_gious)
        print(f"loss_giou: {losses['loss_giou'].item():2f}")
        # take the mean element-wise absolute value difference
        loss_bbox = F.l1_loss(matched_bboxes, target_boxes, reduction="mean")
        losses["loss_bbox"] = loss_bbox
        print(f"loss_bbox: {loss_bbox.item():2f}")
        return losses

    def get_loss(
        self,
        loss,
        outputs,
        targets,
        **kwargs,
    ):
        loss_map = {
            "boxes": self.loss_boxes,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, **kwargs)

    def forward(self, outputs, targets, inter_idx=None, time_mask=None):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the models for the format
             targets: list of dicts, such that len(targets) == n_annotated_frames.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
             inter_idx: list of [start index of the annotated moment, end index of the annotated moment] for each video
             time_mask: [B, T] tensor with False on the padded positions, used to take out padded frames from the loss computation
        """

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(
                self.get_loss(
                    loss,
                    outputs,
                    targets
                )
            )

        return losses


def build(args):
    transformer = build_transformer(args)

    tube_detector = TubeDecoder(
        transformer,
        num_queries=args.num_queries,
        video_max_len=args.video_max_len,
        guided_attn=args.guided_attn,
        sted=False,
    )
    # if args.guided_attn:
    #     weight_dict["loss_guided_attn"] = args.guided_attn_loss_coef

    losses = ["boxes", "sted"] if args.sted else ["boxes"]
    if args.guided_attn:
        losses += ["guided_attn"]

    criterion = SetCriterion(
        losses=losses,
        sigma=args.sigma,
    )
    # criterion = nn.BCELoss()

    return tube_detector, criterion