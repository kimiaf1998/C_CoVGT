# Adapted from
# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
TubeDETR model and criterion classes.
"""
from typing import Dict, Optional

import torch
import torch.distributed
import torch.nn.functional as F
from torch import nn
import math

from tools import box_ops, dist
# from util import box_ops

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
        :param transformer: transformer model
        :param num_queries: number of object queries per frame
        :param video_max_len: maximum number of frames in the model
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

        print("hs shape:", hs.shape)
        # hs = hs.flatten(1, 2)  # n_layersx(b*t)xnum_queriesxf

        print("pred_boxes hs : ", hs[-1][0,:,0])
        outputs_coord = self.bbox_embed(hs).sigmoid() # n_layersx(b*t)xnum_queriesx1
        out.update({"pred_boxes": outputs_coord[-1]}) # fetch last-layer output ->  (b*t)xnum_queriesx1
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

    def loss_boxes(self, outputs, targets, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        src_boxes = outputs["pred_boxes"]
        target_boxes = torch.cat([t["bboxes"] for t in targets], dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / max(num_boxes, 1)

        # TODO (no need for xyxy conversion if already are)
        # TODO check if dialog is a valid way of comparing
        loss_giou = 1 - box_ops.generalized_box_iou(
                src_boxes,
                target_boxes,
            )
        losses["loss_giou"] = loss_giou.sum() / max(num_boxes, 1)
        return losses

    def loss_sted(self, outputs, num_boxes, inter_idx, positive_map, time_mask=None):
        """Compute the losses related to the start & end prediction, a KL divergence loss
        targets dicts must contain the key "pred_sted" containing a tensor of logits of dim [T, 2]
        """
        assert "pred_sted" in outputs
        sted = outputs["pred_sted"]
        losses = {}

        target_start = torch.tensor([x[0] for x in inter_idx], dtype=torch.long).to(
            sted.device
        )
        target_end = torch.tensor([x[1] for x in inter_idx], dtype=torch.long).to(
            sted.device
        )
        sted = sted.masked_fill(
            ~time_mask[:, :, None], -1e32
        )  # put very low probability on the padded positions before softmax
        eps = 1e-6  # avoid log(0) and division by 0

        sigma = self.sigma
        start_distrib = (
            -(
                (
                    torch.arange(sted.shape[1])[None, :].to(sted.device)
                    - target_start[:, None]
                )
                ** 2
            )
            / (2 * sigma ** 2)
        ).exp()  # gaussian target
        start_distrib = F.normalize(start_distrib + eps, p=1, dim=1)
        pred_start_prob = (sted[:, :, 0]).softmax(1)
        loss_start = (
            pred_start_prob * ((pred_start_prob + eps) / start_distrib).log()
        )  # KL div loss
        loss_start = loss_start * time_mask  # not count padded values in the loss

        end_distrib = (
            -(
                (
                    torch.arange(sted.shape[1])[None, :].to(sted.device)
                    - target_end[:, None]
                )
                ** 2
            )
            / (2 * sigma ** 2)
        ).exp()  # gaussian target
        end_distrib = F.normalize(end_distrib + eps, p=1, dim=1)
        pred_end_prob = (sted[:, :, 1]).softmax(1)
        loss_end = (
            pred_end_prob * ((pred_end_prob + eps) / end_distrib).log()
        )  # KL div loss
        loss_end = loss_end * time_mask  # do not count padded values in the loss

        loss_sted = loss_start + loss_end
        losses["loss_sted"] = loss_sted.mean()

        return losses

    def loss_guided_attn(
        self, outputs, num_boxes, inter_idx, positive_map, time_mask=None
    ):
        """Compute guided attention loss
        targets dicts must contain the key "weights" containing a tensor of attention matrices of dim [B, T, T]
        """
        weights = outputs["weights"]  # BxTxT

        positive_map = positive_map + (
            ~time_mask
        )  # the padded positions also have to be taken out
        eps = 1e-6  # avoid log(0) and division by 0

        loss = -(1 - weights + eps).log()
        loss = loss.masked_fill(positive_map[:, :, None], 0)
        nb_neg = (~positive_map).sum(1) + eps
        loss = loss.sum(2) / nb_neg[:, None]  # sum on the column
        loss = loss.sum(1)  # mean on the line normalized by the number of negatives
        loss = loss.mean()  # mean on the batch

        losses = {"loss_guided_attn": loss}
        return losses

    def get_loss(
        self,
        loss,
        outputs,
        targets,
        num_boxes,
        inter_idx,
        positive_map,
        time_mask,
        **kwargs,
    ):
        loss_map = {
            "boxes": self.loss_boxes,
            "sted": self.loss_sted,
            "guided_attn": self.loss_guided_attn,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        if loss in ["sted", "guided_attn"]:
            return loss_map[loss](
                outputs, num_boxes, inter_idx, positive_map, time_mask, **kwargs
            )
        return loss_map[loss](outputs, targets, num_boxes, **kwargs)

    def forward(self, outputs, targets, inter_idx=None, time_mask=None):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == n_annotated_frames.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
             inter_idx: list of [start index of the annotated moment, end index of the annotated moment] for each video
             time_mask: [B, T] tensor with False on the padded positions, used to take out padded frames from the loss computation
        """
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["bboxes"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if dist.is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / dist.get_world_size(), min=1).item()

        if inter_idx is not None and time_mask is not None:
            # construct a map such that positive_map[k, i] = True iff num_frame i lies inside the annotated moment k
            positive_map = torch.zeros(time_mask.shape, dtype=torch.bool)
            for k, idx in enumerate(inter_idx):
                if idx[0] < 0:  # empty intersection
                    continue
                positive_map[k][idx[0] : idx[1] + 1].fill_(True)

            positive_map = positive_map.to(time_mask.device)
        elif time_mask is None:
            positive_map = None

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(
                self.get_loss(
                    loss,
                    outputs,
                    targets,
                    num_boxes,
                    inter_idx,
                    positive_map,
                    time_mask,
                )
            )

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                for loss in self.losses:
                    kwargs = {}
                    l_dict = self.get_loss(
                        loss,
                        aux_outputs,
                        targets,
                        num_boxes,
                        inter_idx,
                        positive_map,
                        time_mask,
                        **kwargs,
                    )
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


def build(args):
    transformer = build_transformer(args)

    tube_detector = TubeDecoder(
        transformer,
        num_queries=args.num_queries,
        video_max_len=args.video_max_len,
        guided_attn=args.guided_attn,
        sted=True,
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