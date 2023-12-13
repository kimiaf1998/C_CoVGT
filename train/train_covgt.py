import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import collections
from util import compute_aggreeings, AverageMeter, get_mask, mask_tokens
import os.path as osp
import json
from tqdm import tqdm
#from fvcore.nn import FlopCountAnalysis
from tools.postprocess import PostProcess
from eval.star_eval import STAREvaluator

def eval(model, data_loader, a2v, args, test=False, tokenizer="RoBERTa"):
    model.eval()
    count = 0
    metrics, counts = collections.defaultdict(float), collections.defaultdict(int)

    qa_predictions = {}
    loc_predictions = {}

    outputs = {"results": {},
               "metrics": {}}
    print("** Evaluating **")

    with torch.no_grad():
        if not args.mc:
            model.module._compute_answer_embedding(a2v)
        for i, batch in enumerate(tqdm(data_loader, desc="Evaluating batches", unit="batch")):
            if i == 2:
                break
            answer_id, answer, video_o, video_f, vid_orig_size, question, question_id, seg_feats, seg_num , bboxes, bboxes_mask = (
                batch["answer_id"],
                batch["answer"].cuda(),
                batch["video_o"].cuda(),
                batch["video_f"].cuda(),
                batch["orig_size"],
                batch["question"].cuda(),
                batch['question_id'],
                batch['seg_feats'].cuda(),
                batch['seg_num'],
                batch['bboxes'],  # visual answer locations (gt)
                batch['bboxes_mask']  # mask of visual answer locations (gt)
            )

            video_len = batch["video_len"]
            max_object_len = max(batch["object_len"])
            seq_len = batch["seq_len"]

            question_mask = (question!=tokenizer.pad_token_id).float() #RobBERETa
            answer_mask = (answer!=tokenizer.pad_token_id).float() #RobBERETa

            bs, numc, numf, max_object_num, _ = video_o.size()
            video_mask = get_mask(video_len, video_o.size(1)).cuda()
            object_mask = (torch.arange(max_object_len).unsqueeze(1).to(video_o.device) < video_o.size(2)).repeat(1, max_object_len)

            count += answer_id.size(0)
            video = (video_o, video_f)
            if not args.mc:
                predicts, tube_pred = model(
                    video,
                    question,
                    text_mask=question_mask,
                    video_mask=video_mask,
                    object_mask=object_mask,
                    seq_len = seq_len,
                    localization=True,
                )
                topk = torch.topk(predicts, dim=1, k=10).indices.cpu()
                if args.dataset != "ivqa":
                    answer_id_expanded = answer_id.view(-1, 1).expand_as(topk)
                else:
                    answer_id = (answer_id / 2).clamp(max=1)
                    answer_id_expanded = answer_id
                # TODO divide the metrics res by count
                metrics = compute_aggreeings(
                    topk,
                    answer_id_expanded,
                    [1, 10],
                    ["acc", "acc10"],
                    metrics,
                    ivqa=(args.dataset == "ivqa"),
                )
                for bs, qid in enumerate(question_id):
                    qa_predictions[qid] = {'prediction': int(topk.numpy()[bs,0]), 'answer':int(answer_id.numpy()[bs])}
            else:
                #############Model FLOPs##########
                # inputs = (video, question, None, answer.cuda(), seq_len, video_mask, answer_mask)
                # flops = FlopCountAnalysis(models, inputs)
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
                    seg_num = seg_num,
                    localization = True,
                )

                fusion_proj = fusion_proj.unsqueeze(2)
                predicts = torch.bmm(answer_proj, fusion_proj).squeeze()

                predicted = torch.max(predicts, dim=1).indices.cpu()
                # calculate textual answer accuracy
                metrics["acc"] += (predicted == answer_id).sum().item()
                # choices are returned in the format of a list of mcx(bs tuples).
                choices = batch["choices"]
                questions = batch["question_txt"]
                for bs, qid in enumerate(question_id):
                    question = questions[bs]
                    pred_id = int(predicted.numpy()[bs])
                    ans_id = int(answer_id.numpy()[bs])
                    pred = choices[pred_id][bs]
                    ans = choices[ans_id][bs]
                    qa_predictions[qid] = {'question': question, 'prediction': pred, 'answer':ans}


                # convert predicts from relative [0, 1] to absolute [0, height] coordinates
                # results = PostProcess()(tube_pred["pred_boxes"], vid_orig_size) # TODO load orig_size (needs maximum object finding among 10)
                # tube_pred["pred_boxes"] = tube_pred["pred_boxes"].reshape(bs, (numc*numf), max_object_num, -1)
                evaluator = STAREvaluator(targets=batch, save_pred=True)
                bs, numc, numf, _, _ = video_o.size()
                tube_pred["pred_boxes"] = tube_pred["pred_boxes"].reshape(bs, (numc * numf), args.num_queries,
                                                                          -1)  # (bs*t)xnum_queriesx1 -> bsxtxnum_queriesx4
                evaluator.update(tube_pred["pred_boxes"])
                loc_output = evaluator.summarize()
                loc_predictions.update(loc_output["predictions"])
                loc_output.pop("predictions")

                for k, v in loc_output.items():
                    if k in outputs["metrics"]:
                        prev_val = outputs["metrics"][k]
                        new_val = (v + prev_val) / 2
                    else:
                        new_val = v
                    outputs["metrics"].update({k: new_val})

    # merge qa + localization results
    outputs["metrics"].update({"acc": metrics["acc"] / count})
    outputs["results"] = {
        question_id: {
            "prediction": {"desc": qa_predictions[question_id]['prediction'],
                           "box": loc_predictions[question_id]['prediction'].detach().cpu().tolist()},
            "answer": {"desc": qa_predictions[question_id]['answer'],
                       "box": loc_predictions[question_id]['answer'].detach().cpu().tolist()}
        }
        for question_id in loc_predictions.keys() # just going with annotated samples (having bbox)
        }

    step = "val" if not test else "test"
    for k, v in outputs["metrics"].items():
        print(f"{step} {k}: {v:.2%}")
        logging.info(f"{step} {k}: {v:.2%}")

    return outputs


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
    for i, batch in enumerate(tqdm(train_loader, desc="Training on batches", unit="batch")):
        if i == 2:
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
            batch['bboxes'].cuda()        # visual answer locations
        )

        video_len = batch["video_len"]
        max_object_len = max(batch["object_len"])

        question_mask = (question != tokenizer.pad_token_id).float().cuda() #RobBERETa
        answer_mask = (answer!=tokenizer.pad_token_id).float().cuda() #RobBERETa
        video_mask = get_mask(video_len, video_o.size(1)).cuda()
        object_mask = (torch.arange(max_object_len).unsqueeze(1).to(video_o.device) < video_o.size(2)).repeat(1, max_object_len)

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
                seq_len = seq_len,
                localization=True,
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
                seg_num = seg_num,
                localization=True,
            )   # outputs video and answer representation

            fusion_proj = fusion_proj.unsqueeze(2)
            # Calculates dot-product or video and answer repr. to find the best match
            predicts = torch.bmm(answer_proj, fusion_proj).squeeze()

        # only keep box predictions in the annotated moment
        device = tube_pred["pred_boxes"].device

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
                bs, numc, numf, _, _ = video_o.size()
                tube_pred["pred_boxes"] = tube_pred["pred_boxes"].reshape(bs, (numc * numf), args.num_queries,
                                                                          -1)  # (bs*t)xnum_queriesx1 -> bsxtxnum_queriesx4
                loss_dict.update(loc_criterion(tube_pred, batch))

        loss_dict.update({"loss_vqa": vqa_loss})

        if args.cl_loss:
            vt_proj, txt_proj, _ = model(
                video,
                question,
                text_mask=qsn_mask,
                video_mask=video_mask,
                object_mask=object_mask,
                answer=qsn_token_ids,
                seq_len = qsn_seq_len,
                seg_feats = seg_feats,
                seg_num = seg_num,
            )
            vt_proj = vt_proj.unsqueeze(2)
            cl_predicts = torch.bmm(txt_proj, vt_proj).squeeze()
            cl_loss = qa_criterion(cl_predicts, qsn_id.cuda())
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
        # sted_loss = loss_dict["loss_sted"]
        losses = sum(
            loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
        )
        print("Total loss:", loss)

        optimizer.zero_grad()
        losses.backward()
        if args.clip:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip)
        optimizer.step()
        scheduler.step()

        # Access the updated learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Print the updated learning rate
        print(f'Learning Rate: {current_lr}')

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
