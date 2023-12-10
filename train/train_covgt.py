import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import collections
from util import compute_aggreeings, AverageMeter, get_mask, mask_tokens
import os.path as osp
import json
#from fvcore.nn import FlopCountAnalysis
from tools.postprocess import PostProcess
from eval.star_eval import STAREvaluator

def eval(model, data_loader, a2v, args, test=False, tokenizer="RoBERTa"):
    model.eval()
    count = 0
    metrics, counts = collections.defaultdict(int), collections.defaultdict(int)

    with torch.no_grad():
        if not args.mc:
            model.module._compute_answer_embedding(a2v)
        results = {}
        for i, batch in enumerate(data_loader):
            if i==2:
                break
            answer_id, answer, video_o, video_f, vid_orig_size, question, question_id, seg_feats, seg_num , bboxes = (
                batch["answer_id"],
                batch["answer"].cuda(),
                batch["video_o"].cuda(),
                batch["video_f"].cuda(),
                batch["orig_size"],
                batch["question"].cuda(),
                batch['question_id'],
                batch['seg_feats'].cuda(),
                batch['seg_num'],
                batch['bboxes']  # visual answer locations
            )

            video_len = batch["video_len"]
            max_object_len = max(batch["object_len"])
            seq_len = batch["seq_len"]

            question_mask = (question!=tokenizer.pad_token_id).float() #RobBERETa
            answer_mask = (answer!=tokenizer.pad_token_id).float() #RobBERETa

            print("video_o shape:", video_o.size())
            bs, _, _, max_object_num, _ = video_o.size()
            video_mask = get_mask(video_len, video_o.size(1)).cuda()
            print("video_mask:", video_mask.size())
            # object_mask = get_mask(object_len.flatten(0,1), max_object_num).bool().cuda()
            print("object_len:", max_object_len)
            object_mask = (torch.arange(max_object_len).unsqueeze(1).to(video_o.device) < video_o.size(2)).repeat(1, max_object_len)
            print("object_mask size:", object_mask.size())
            count += answer_id.size(0)
            video = (video_o, video_f)
            if not args.mc:
                predicts, tube_pred = model(
                    video,
                    question,
                    text_mask=question_mask,
                    video_mask=video_mask,
                    object_mask=object_mask,
                    seq_len = seq_len
                )
                topk = torch.topk(predicts, dim=1, k=10).indices.cpu()
                if args.dataset != "ivqa":
                    answer_id_expanded = answer_id.view(-1, 1).expand_as(topk)
                else:
                    answer_id = (answer_id / 2).clamp(max=1)
                    answer_id_expanded = answer_id
                metrics = compute_aggreeings(
                    topk,
                    answer_id_expanded,
                    [1, 10],
                    ["acc", "acc10"],
                    metrics,
                    ivqa=(args.dataset == "ivqa"),
                )
                for bs, qid in enumerate(question_id):
                    results[qid] = {'prediction': int(topk.numpy()[bs,0]), 'answer':int(answer_id.numpy()[bs])}
            else:
                #############Model FLOPs##########
                # inputs = (video, question, None, answer.cuda(), seq_len, video_mask, answer_mask)
                # flops = FlopCountAnalysis(model, inputs)
                # print('Model FLOPs:', flops.total()/1000000) #use batch_size 1
                # break
                ###################################
                fusion_proj, answer_proj, tube_pred = model(
                    video,
                    question,
                    text_mask=answer_mask,
                    video_mask=video_mask,
                    object_mask=object_mask,
                    answer=answer,
                    seq_len = seq_len,
                    seg_feats = seg_feats,
                    seg_num = seg_num
                )
                # predicts = fusion_proj.squeeze()

                fusion_proj = fusion_proj.unsqueeze(2)
                predicts = torch.bmm(answer_proj, fusion_proj).squeeze()

                predicted = torch.max(predicts, dim=1).indices.cpu()
                # calculate textual answer accuracy
                metrics["acc"] += (predicted == answer_id).sum().item()
                for bs, qid in enumerate(question_id):
                    results[qid] = {'prediction': int(predicted.numpy()[bs]), 'answer':int(answer_id.numpy()[bs])}


            # convert predicts from relative [0, 1] to absolute [0, height] coordinates
            # results = PostProcess()(tube_pred["pred_boxes"], vid_orig_size) # TODO load orig_size (needs maximum object finding among 10)
            evaluator = STAREvaluator(targets=bboxes, frame_mapping=batch["frame_mapping"])
            evaluator.update(tube_pred["pred_boxes"])
            evaluator.summarize()




    step = "val" if not test else "test"

    for k in metrics:
        # print(metrics[k], count)
        v = metrics[k] / count
        logging.info(f"{step} {k}: {v:.2%}")
        break

    return metrics["acc"] / count, results

def bbox_iou(box1, box2):

    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters:
    box1 : tensor of bounding boxes, shape (N, 4)
    box2 : tensor of bounding boxes, shape (M, 4)

    Returns:
    iou : tensor of IoU values, shape (N, M)
    """
    # Calculate the coordinates of the intersection rectangles
    x_left = torch.max(box1[:, None, 0], box2[:, 0])
    y_top = torch.max(box1[:, None, 1], box2[:, 1])
    x_right = torch.min(box1[:, None, 2], box2[:, 2])
    y_bottom = torch.min(box1[:, None, 3], box2[:, 3])

    # Calculate intersection area
    intersection_area = (x_right - x_left).clamp(min=0) * (y_bottom - y_top).clamp(min=0)

    # Calculate the area of both bounding boxes
    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    # Calculate union area
    union_area = box1_area[:, None] + box2_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area

    return iou

def train(model, train_loader, a2v, optimizer, qa_criterion, loc_criterion, weight_dict, scheduler, epoch, args, tokenizer):
    model.train()
    running_vqa_loss, running_acc, running_mlm_loss, running_cl_loss, running_bbox_loss, running_giou_loss, running_sted_loss = (
        AverageMeter(),
        AverageMeter(),
        AverageMeter(),
        AverageMeter(),
        AverageMeter(),
        AverageMeter(),
        AverageMeter()
    )
    for i, batch in enumerate(train_loader):
        if i == 5:
            break
        answer_id, answer, video_o, video_b, video_f, question, seg_feats, seg_num, qsn_id, qsn_token_ids, qsn_seq_len, bboxes = (
            batch["answer_id"],         # answer id among a choices
            batch["answer"],            # answers (qsn + answers (choices)) token ids
            batch["video_o"].cuda(),    # region feats
            batch["video_b"].cuda(),    # region bboxes
            batch["video_f"].cuda(),    # appearance feats
            batch["question"].cuda(),   # qsns embeddings (open-ended)
            batch['seg_feats'].cuda(),  # answers feats (mc, amax words, 2048)
            batch['seg_num'],           # mc
            batch['qsn_id'],            # qsn id among q choice (used for contrastive learning)
            batch['qsn_token_ids'],     # qsns token ids
            batch['qsn_seq_len'],        # length of qsns token ids
            batch['bboxes']        # visual answer locations
        )

        device = video_b.device

        assert len(bboxes) == len(video_b)
        video_b = video_b.flatten(1, 2)
        print("max bbox x,y:", bboxes.max())
        print("GT box shape:", bboxes.shape)
        print("GT boxes:", bboxes)
        print("orig size:", batch["orig_size"][0])
        print("Feat boxes:", video_b)
        continue

        # for every boxes in every frames
        from torchvision.ops import complete_box_iou, box_iou
        for vid_boxes, vid_gt in zip(video_b.flatten(1,2)[:3], bboxes.to(device)[:3]):
            for f_bboxes, f_gt in zip(vid_boxes, vid_gt):
                print("feature bboxes:", f_bboxes.shape)
                print("gt bboxes:", f_gt.shape)
                # iou = complete_box_iou(f_gt, f_bboxes)
                iou = box_iou(f_gt, f_bboxes)
                print("IoU Res", iou)
        continue


        video_len = batch["video_len"]
        max_object_len = max(batch["object_len"])

        question_mask = (question != tokenizer.pad_token_id).float().cuda() #RobBERETa
        answer_mask = (answer!=tokenizer.pad_token_id).float().cuda() #RobBERETa
        video_mask = (
            get_mask(video_len, video_o.size(1)).cuda() if args.video_max_len > 0 else None
        )
        print("video_o shape:", video_o.size())
        video_mask = get_mask(video_len, video_o.size(1)).cuda()
        print("video_mask:", video_mask.size())
        # object_mask = get_mask(video_o.size(2), object_len).bool().cuda()
        object_mask = (torch.arange(max_object_len).unsqueeze(1).to(video_o.device) < video_o.size(2)).repeat(1, max_object_len)
        print("object_mask size:", object_mask.size())

        qsn_mask = (qsn_token_ids != tokenizer.pad_token_id).float().cuda()

        video = (video_o, video_f)
        N = answer_id.size(0)
        seq_len = batch["seq_len"]  # length of answers token ids
        if not args.mc:
            model.module._compute_answer_embedding(a2v)
            predicts, tube_pred = model(
                video,
                question,
                text_mask=question_mask,
                video_mask=video_mask,
                object_mask=object_mask,
                seq_len = seq_len
            )
        else:
            fusion_proj, answer_proj, tube_pred = model(
                video,
                question,
                text_mask=answer_mask,
                video_mask=video_mask,
                object_mask=object_mask,
                answer=answer.cuda(),
                seq_len = seq_len,
                seg_feats = seg_feats,
                seg_num = seg_num
            )   # outputs video and answer representation

            fusion_proj = fusion_proj.unsqueeze(2)
            # Calculates dot-product or video and answer repr. to find the best match
            predicts = torch.bmm(answer_proj, fusion_proj).squeeze()

        print("** Processing Tube Predictions")
        # only keep box predictions in the annotated moment
        device = tube_pred["pred_boxes"].device
        # inter_idx = batch["inter_idx"]
        # frame_mapping = batch["frame_map"]
        # keep_list = []
        # TODO substitute this with bbox masks (keep the ones that mask == 1)
        # for i_dur, inter in enumerate(inter_idx):
        #     start_t, end_t = inter[0], inter[1]
        #     # TODO find closest frame in frame_mapping
        #     start_t_mapped = frame_mapping[i_dur][start_t]
        #     end_t_mapped = frame_mapping[i_dur][end_t]
        #     keep_list.extend(
        #         [
        #             elt
        #             for elt in range(
        #             start_t_mapped,
        #             end_t_mapped + 1,
        #         )
        #         ]
        #     )
        # keep = torch.tensor(keep_list).long().to(device)
        # print("keep indices: ", keep)
        # TODO maybe keep all the boxes and base o bboxes predict time
        print("pred_boxes before:", tube_pred["pred_boxes"].shape)
        # tube_pred["pred_boxes"] = tube_pred["pred_boxes"][keep]
        # print("pred_boxes after:", tube_pred["pred_boxes"].shape)
        # for i_aux in range(len(outputs["aux_outputs"])):
        #     outputs["aux_outputs"][i_aux]["pred_boxes"] = outputs["aux_outputs"][i_aux][
        #         "pred_boxes"
        #     ][keep]
        # b = len(durations)

        # bboxes -> (bs, t, 1, 4)

        empty_box = [0,0,0,0]
        print('All target bbox len batch 1:', len(bboxes[0]))

        # Extract the last elements along the last dimension
        # last_elements = bboxes[:, :, :, -4:]

        # Check if the last elements are [0, 0, 0, 0]
        # import numpy as np
        # condition = np.all(last_elements == empty_box, axis=-1)

        # keep only targets in the annotated moments
        # bboxes = bboxes[condition]
        # print('Non-empty target bbox len batch 1:', len(bboxes[0]))

        print(tube_pred["pred_boxes"][0, :, 0])

        # tube_pred["pred_boxes"] -> (bs*t)xnum_queriesx1
        # bboxes -> bsxtxnum_queriesx4

        # bs = len(video_len)
        # tube_pred["pred_boxes"] = tube_pred["pred_boxes"].reshape(bs, -1, max_object_len, 1)

        # print(tube_pred["pred_boxes"][0, 0, :, 0])

        # get 4 positions of each predicted bbox as True
        # Use boolean indexing to select the bounding boxes
        # video_b = video_b.flatten(1,2)
        # selected_bboxes = torch.zeros(video_b.shape).to(device)  # Initialize tensor for selected bounding boxes
        # print("bbox app feat shape:", video_f.shape)
        # print("bbox region feat shape:", video_o.shape)
        # print("bbox loc shape:", video_b.shape)

        # # TODO maybe adding mask for False objects of check if an empty bbox in loss calculation
        # print("selected_bbox [0][5]", selected_bboxes[0][5])
        # tube_pred["pred_boxes"] = selected_bboxes.clone()

        # compute losses
        loss_dict = {}
        if args.dataset == "ivqa":
            a = (answer_id / 2).clamp(max=1).cuda()
            vqa_loss = qa_criterion(predicts, a)
            predicted = torch.max(predicts, dim=1).indices.cpu()
            predicted = F.one_hot(predicted, num_classes=len(a2v))
            running_acc.update((predicted * a.cpu()).sum().item() / N, N)
        else:
            vqa_loss = qa_criterion(predicts, answer_id.cuda())
            predicted = torch.max(predicts, dim=1).indices.cpu()
            running_acc.update((predicted == answer_id).sum().item() / N, N)


            # mask with padded positions set to False for loss computation
            # TODO
            if args.sted:
                time_mask = torch.zeros(b, outputs["pred_sted"].shape[1]).bool().to(device)
                for i_dur, duration in enumerate(durations):
                    time_mask[i_dur, :duration] = True
            else:
                time_mask = None

            if loc_criterion is not None:
                loss_dict.update(loc_criterion(tube_pred["pred_boxes"], targets))

        loss_dict.update({"loss_vqa": vqa_loss})

        if args.cl_loss:
            vt_proj, txt_proj = model(
                video,
                question,
                text_mask=qsn_mask,
                video_mask=video_mask,
                object_mask=object_mask,
                answer=qsn_token_ids,
                seq_len = qsn_seq_len,
                seg_feats = seg_feats,
                seg_num = seg_num
            )
            vt_proj = vt_proj.unsqueeze(2)
            cl_predicts = torch.bmm(txt_proj, vt_proj).squeeze()
            cl_loss = criterion(cl_predicts, qsn_id.cuda())
            loss_dict.update({"loss_cl": args.cl_loss*cl_loss})
            # cl_predicted = torch.max(cl_predicts, dim=1).indices.cpu()
            # running_acc.update((predicted == answer_id).sum().item() / N, N)

        if args.mlm_prob:
            max_seq_len = args.qmax_words
            if args.mc > 0:
                tmp_id = [aid+(args.mc*i) for i, aid in enumerate(answer_id)]
                inputs = answer.view(N*args.mc, -1)[tmp_id,:]
                # question_mask = (inputs>0).float()
                question_mask = (inputs!=1).float()
                max_seq_len = args.amax_words
            else:
                inputs = batch["question"]

            inputs, labels = mask_tokens(inputs, tokenizer, mlm_probability=args.mlm_prob)
            mlm_loss = model(
                video,
                question=inputs.cuda(),
                labels=labels.cuda(),
                text_mask=question_mask,
                video_mask=video_mask,
                object_mask=object_mask,
                max_seq_len=max_seq_len,
                mode="mlm",
            )
            loss_dict.update({"loss_mlm": mlm_loss})
            mlm_loss = mlm_loss.mean()
            if not args.cl_loss:
                loss = mlm_loss + vqa_loss
                loss_dict.update({"loss_mlm": loss})

        bbox_loss = loss_dict["loss_bbox"]
        giou_loss = loss_dict["loss_giou"]
        sted_loss = loss_dict["loss_sted"]
        losses = sum(
            loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
        )

        optimizer.zero_grad()
        losses.backward()
        if args.clip:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip)
        optimizer.step()
        scheduler.step()

        running_vqa_loss.update(vqa_loss.detach().cpu().item(), N)
        running_giou_loss.update(giou_loss.detach().cpu().item(), N)
        running_bbox_loss.update(bbox_loss.detach().cpu().item(), N)
        if args.sted:
            running_sted_loss.update(sted_loss.detach().cpu().item(), N)
        if args.mlm_prob:
            running_mlm_loss.update(mlm_loss.detach().cpu().item(), N)
        if args.cl_loss:
            running_cl_loss.update(cl_loss.detach().cpu().item(), N)
        if (i + 1) % (len(train_loader) // args.freq_display) == 0:
            log = "Epoch {epoch + 1}/{args.epochs}, Progress: {float(i + 1) / len(train_loader):.4f}, Lvqa loss: "\
                    f"{running_vqa_loss.avg:.4f}, Train acc: {running_acc.avg:.2%}"
            if args.mlm_prob:
                log += f", MLM loss: {running_mlm_loss.avg:.4f}"
            elif args.cl_loss:
                log += f", Lvq Loss: {running_cl_loss.avg:.4f}"
            elif args.sted:
                log += f", STED Loss: {running_sted_loss.avg:.4f}"
            log += f", BBox L1 Loss: {running_bbox_loss.avg:.4f}, gIoU Loss: {running_giou_loss.avg:.4f} "
            logging.info(log
            )
            running_acc.reset()
            running_vqa_loss.reset()
            running_mlm_loss.reset()
            running_cl_loss.reset()
