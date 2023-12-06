import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import collections
from util import compute_aggreeings, AverageMeter, get_mask, mask_tokens
import os.path as osp
import json
#from fvcore.nn import FlopCountAnalysis

def eval(model, data_loader, a2v, args, test=False, tokenizer="RoBERTa"):
    model.eval()
    count = 0
    metrics, counts = collections.defaultdict(int), collections.defaultdict(int)

    with torch.no_grad():
        if not args.mc:
            model.module._compute_answer_embedding(a2v)
        results = {}
        for i, batch in enumerate(data_loader):
            answer_id, answer, video_o, video_f, question, question_id, seg_feats, seg_num = (
                batch["answer_id"],
                batch["answer"].cuda(),
                batch["video_o"].cuda(),
                batch["video_f"].cuda(),
                batch["question"].cuda(),
                batch['question_id'],
                batch['seg_feats'].cuda(),
                batch['seg_num']
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
                # TODO get tube output
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
                metrics["acc"] += (predicted == answer_id).sum().item()
                for bs, qid in enumerate(question_id):
                    results[qid] = {'prediction': int(predicted.numpy()[bs]), 'answer':int(answer_id.numpy()[bs])}

    step = "val" if not test else "test"

    for k in metrics:
        # print(metrics[k], count)
        v = metrics[k] / count
        logging.info(f"{step} {k}: {v:.2%}")
        break

    return metrics["acc"] / count, results


def train(model, train_loader, a2v, optimizer, criterion, scheduler, epoch, args, tokenizer):
    model.train()
    running_vqa_loss, running_acc, running_mlm_loss, running_cl_loss = (
        AverageMeter(),
        AverageMeter(),
        AverageMeter(),
        AverageMeter()
    )
    for i, batch in enumerate(train_loader):
        answer_id, answer, video_o, video_f, question, seg_feats, seg_num, qsn_id, qsn_token_ids, qsn_seq_len, bboxes = (
            batch["answer_id"],         # answer id among a choices
            batch["answer"],            # answers (qsn + answers (choices)) token ids
            batch["video_o"].cuda(),    # region feats
            batch["video_f"].cuda(),    # appearance feats
            batch["question"].cuda(),   # qsns embeddings (open-ended)
            batch['seg_feats'].cuda(),  # answers feats (mc, amax words, 2048)
            batch['seg_num'],           # mc
            batch['qsn_id'],            # qsn id among q choice (used for contrastive learning)
            batch['qsn_token_ids'],     # qsns token ids
            batch['qsn_seq_len'],        # length of qsns token ids
            batch['bboxes']        # visual answer locations
        )
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
        inter_idx = batch["inter_idx"]
        frame_mapping = batch["frame_map"]
        keep_list = []
        for i_dur, inter in enumerate(inter_idx):
            start_t, end_t = inter[0], inter[1]
            start_t_mapped = frame_mapping[i_dur][start_t]
            end_t_mapped = frame_mapping[i_dur][end_t]
            keep_list.extend(
                [
                    elt
                    for elt in range(
                    start_t_mapped,
                    end_t_mapped + 1,
                )
                ]
            )
        keep = torch.tensor(keep_list).long().to(device)
        print("keep indices: ", keep)
        # TODO maybe keep all the boxes and base o bboxes predict time
        print("pred_boxes before:", tube_pred["pred_boxes"].shape)
        # tube_pred["pred_boxes"] = tube_pred["pred_boxes"][keep]
        # print("pred_boxes after:", tube_pred["pred_boxes"].shape)
        # for i_aux in range(len(outputs["aux_outputs"])):
        #     outputs["aux_outputs"][i_aux]["pred_boxes"] = outputs["aux_outputs"][i_aux][
        #         "pred_boxes"
        #     ][keep]
        # b = len(durations)

        # targets = [
        #     x for x in bboxes if len(x["boxes"])
        # ]  # keep only targets in the annotated moment

        # tube_pred["pred_boxes"] -> (bs*t)xnum_queriesx1
        # bboxes -> bsxtxnum_queriesx4

        bs = len(batch)
        t = tube_pred["pred_boxes"].size(0)//bs
        tube_pred["pred_boxes"] = tube_pred["pred_boxes"].reshape(bs, t, -1, 1)

        # get 4 positions of each predicted bbox as True
        # Use boolean indexing to select the bounding boxes
        selected_bboxes = torch.zeros(bboxes.shape)  # Initialize tensor for selected bounding boxes

        for i in range(bs):
            for j in range(t):
                # Get indices of True values in bboxes_pred for each batch and time step
                true_indices = torch.where(tube_pred["pred_boxes"][i, j, :, 0])[0]
                if true_indices.numel() > 0:
                    # Assign the corresponding bboxes to selected_bboxes
                    selected_bboxes[i, j, true_indices] = bboxes[i, j, true_indices]

        print("selected_bbox [0][5]", selected_bboxes[0][5])

        assert len(selected_bboxes) == len(tube_pred["pred_boxes"]), (
            len(tube_pred["pred_boxes"]),
            len(bboxes),
        )

        # TODO

        if args.dataset == "ivqa":
            a = (answer_id / 2).clamp(max=1).cuda()
            vqa_loss = criterion(predicts, a)
            predicted = torch.max(predicts, dim=1).indices.cpu()
            predicted = F.one_hot(predicted, num_classes=len(a2v))
            running_acc.update((predicted * a.cpu()).sum().item() / N, N)
        else:
            vqa_loss = criterion(predicts, answer_id.cuda())
            predicted = torch.max(predicts, dim=1).indices.cpu()
            running_acc.update((predicted == answer_id).sum().item() / N, N)
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
            mlm_loss = mlm_loss.mean()
            loss = mlm_loss + vqa_loss
        if args.cl_loss:
            loss = vqa_loss + args.cl_loss*cl_loss
        if args.cl_loss and args.mlm_prob:
            loss = vqa_loss + args.cl_loss*cl_loss + mlm_loss
        if not args.cl_loss and not args.mlm_prob:
            loss = vqa_loss

        optimizer.zero_grad()
        loss.backward()
        if args.clip:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip)
        optimizer.step()
        scheduler.step()

        running_vqa_loss.update(vqa_loss.detach().cpu().item(), N)
        if args.mlm_prob:
            running_mlm_loss.update(mlm_loss.detach().cpu().item(), N)
        if args.cl_loss:
            running_cl_loss.update(cl_loss.detach().cpu().item(), N)
        if (i + 1) % (len(train_loader) // args.freq_display) == 0:
            if args.mlm_prob:
                logging.info(
                    f"Epoch {epoch + 1}/{args.epochs}, Progress: {float(i + 1) / len(train_loader):.4f}, Lvqa loss: "
                    f"{running_vqa_loss.avg:.4f}, Training acc: {running_acc.avg:.2%}, MLM loss: {running_mlm_loss.avg:.4f}, Lvq Loss: {running_cl_loss.avg:.4f}"
                )
            elif args.cl_loss:
                logging.info(
                    f"Epoch {epoch + 1}/{args.epochs}, Progress: {float(i + 1) / len(train_loader):.4f}, Lvqa loss: "
                    f"{running_vqa_loss.avg:.4f}, Train acc: {running_acc.avg:.2%}, Lvq Loss: {running_cl_loss.avg:.4f}"
                )
            else:
                logging.info(
                    f"Epoch {epoch + 1}/{args.epochs}, Progress: {float(i + 1) / len(train_loader):.4f}, Lvqa loss: "
                    f"{running_vqa_loss.avg:.4f}, Train acc: {running_acc.avg:.2%}"
                )
            running_acc.reset()
            running_vqa_loss.reset()
            running_mlm_loss.reset()
            running_cl_loss.reset()
