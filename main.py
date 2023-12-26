import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import os.path as osp
import logging

from transformers import get_cosine_schedule_with_warmup
import models.tubedetr as tube_detector
from args import get_args
from models.Tube_CoVGT import build_model
from util import compute_a2v, load_model_by_key, save_to, plot_and_save_epochs_res
from dataloader.cvqa_loader import get_videoqa_loaders
from train.train_covgt import train, eval
from tools.bbox_visualizer import draw_and_save_rects, add_label
from tools.postprocess import PostProcess
from tqdm import tqdm
from tools.box_ops import box_cxcywh_to_xyxy


def main(args):
    if not (os.path.isdir(args.save_dir)):
        os.mkdir(os.path.join(args.save_dir))

    log_path = os.path.join(args.save_dir, "stdout.log")
    if os.path.exists(log_path):
        try:
            os.remove(log_path)
            print(f"Existing '{log_path}' has been removed.")
        except OSError as e:
            print(f"Error: {e}")

    # Configure the root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(message)s",
        filename=os.path.join(args.save_dir, "stdout.log"),
        filemode="a"  # Use "a" for append mode
    )

    # Create a logger instance
    logger = logging.getLogger("MAIN")
    logger.setLevel(logging.INFO)
    main_handler_logger = logging.FileHandler(log_path, encoding="utf-8")
    main_handler_logger.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(message)s")
    )
    logger.addHandler(main_handler_logger)

    logger.info(args)

    # Model
    model, tokenizer = build_model(args)
    model.cuda()
    logging.info("Using {} GPUs".format(torch.cuda.device_count()))

    a2id, id2a, a2v = None, None, None
    if not args.mc:
        a2id, id2a, a2v = compute_a2v(
            vocab_path=args.vocab_path,
            bert_tokenizer=tokenizer,
            amax_words=args.amax_words,
        )
        logging.info(f"Length of Answer Vocabulary: {len(a2id)}")

    weight_dict = {
        "loss_bbox": args.bbox_loss_coef,
        "loss_giou": args.giou_loss_coef,
        "loss_sted": args.sted_loss_coef,
        "loss_vqa": 5,
        "loss_cl": args.cl_loss,
        "loss_mlm": 1,
    }

    # Load pretrain path
    model = nn.DataParallel(model)
    
    if args.pretrain_path != "":
        # models.load_state_dict(torch.load(args.pretrain_path))
        model.load_state_dict(load_model_by_key(model, args.pretrain_path))
        logging.info(f"Loaded checkpoint {args.pretrain_path}")
    logging.info(
        f"Nb of trainable params:{sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    (
        train_loader,
        val_loader,
        test_loader,
    ) = get_videoqa_loaders(args, args.features_path, a2id, tokenizer, test_mode = args.test)

    if args.test:
        logging.info("number of test instances: {}".format(len(test_loader.dataset)))
    else:
        logging.info("number of train instances: {}".format(len(train_loader.dataset)))
        logging.info("number of val instances: {}".format(len(val_loader.dataset)))


    losses = ["boxes", "sted"] if args.sted else ["boxes"]
    if args.guided_attn:
        losses += ["guided_attn"]

    loc_criterion = tube_detector.SetCriterion(
        losses=losses,
        sigma=args.sigma,
    )

    loc_criterion.cuda()

    qa_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    # criterion = MultipleChoiceLoss()
    params_for_optimization = list(p for p in model.parameters() if p.requires_grad)
    optimizer = optim.Adam(
        params_for_optimization, lr=args.lr, weight_decay=args.weight_decay
    )
    qa_criterion.cuda()

    # Training
    if not args.test:
        num_trainings = len(train_loader) * args.epochs
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,0, num_trainings
        )
        logging.info(
            f"Set cosine schedule with {len(train_loader) * args.epochs} iterations"
        )
        # TODO uncomment
        # if args.pretrain_path != "":
        #     outputs = eval(models, val_loader, a2v, args, test=False, tokenizer=tokenizer)  # zero-shot VideoQA
        #     val_iou = outputs["metrics"]["m_viou"]
        #     results = outputs["results"]
        #     save_path = osp.join(args.save_dir, 'val-res0.json')
        #     save_to (save_path, results)
        # val_acc = 42.0
        # best_val_viou = 0 if args.pretrain_path == "" else val_iou
        best_val_viou = 0
        val_acc = 0.0
        best_epoch = 0
        epochs_val = []
        epochs_loss = []
        for epoch in tqdm(range(args.epochs), desc="Training on epoch", unit="epoch"):
            losses_dict = train(model, train_loader, a2v, optimizer, qa_criterion, loc_criterion, weight_dict, scheduler, epoch, args, tokenizer)
            epochs_loss.append(losses_dict)
            outputs = eval(model, val_loader, a2v, args, test=False, tokenizer=tokenizer)
            val_iou = outputs["metrics"]["m_viou"]
            val_acc = outputs["metrics"]["acc"]
            results = outputs["results"]
            epochs_val.append({"m_viou": round(val_iou,2), "acc": round(val_acc,2)})
            if val_iou > best_val_viou:
                best_val_viou = val_iou
                best_epoch = epoch
                torch.save(
                    model.state_dict(), os.path.join(args.save_dir, f'best_model_mviou_{val_iou*100:.2f}_acc_{val_acc*100:.2f}.pth')
                )
                logging.info(f"Best models have been saved in {os.path.join(args.save_dir, f'best_model_mviou_{val_iou*100:.2f}_acc_{val_acc*100:.2f}.pth')}")
                print(f"Best models have been saved in {os.path.join(args.save_dir, f'best_model_mviou_{val_iou*100:.2f}_acc_{val_acc*100:.2f}.pth')}")
                save_path = osp.join(args.save_dir, 'val-res.json')
                save_to(save_path, results)
            if args.dataset == 'webvid': 
                ep_file = os.path.join(args.save_dir, f"e{epoch}.pth")
                torch.save(model.state_dict(), ep_file)
                logging.info('Save to '+ep_file)

        # Fetch validation and loss results
        epochs = range(len(epochs_val))
        epochs_val_items = {}
        epochs_loss_items = {}
        for key in epochs_val[0].keys():
            epochs_val_items.update({key: [d[key] for d in epochs_val]})

        for key in epochs_loss[0].keys():
            epochs_loss_items.update({key: [d[key] for d in epochs_loss]})
        #
        # for epoch in range(args.epochs):
        #     for val_key in epochs_val.keys():
        #         epochs_val_items[val_key].append(epochs_val[epoch][val_key])
        #     for loss_key in epochs_loss.keys():
        #         epochs_loss_items[loss_key].append(epochs_loss[epoch][loss_key])

        # Plot validation and loss results
        for metric, val in epochs_val_items.items():
            plot_and_save_epochs_res(epochs, val, ylabel=metric, save_path=args.save_dir)
        for metric, loss in epochs_loss_items.items():
            plot_and_save_epochs_res(epochs, loss, ylabel=metric, save_path=args.save_dir)

        logging.info(f"Best val models at epoch {best_epoch + 1} with m_viou {best_val_viou*100:.2f} and acc {val_acc*100:.2f}")
        print(f"Best val models at epoch {best_epoch + 1} with m_viou {best_val_viou*100:.2f} and acc {val_acc*100:.2f}")
    else:
        # Evaluate on val (=test) set
        outputs = eval(model, test_loader, a2v, args, test=True, tokenizer=tokenizer)  # zero-shot VideoQA
        results = outputs["results"]
        save_path = osp.join(args.save_dir, 'val-res0.json')
        save_to(save_path, results)
        # Visualize results
        batch0 = next(iter(test_loader))
        vid_orig_size = batch0["orig_size"][0]
        print("orig size:", vid_orig_size)

        for q_id, values in results.items():

            preds = values["prediction"]
            gt = values["answer"]
            video_id = values['video_id']
            video_frame_ids = values['frame_mapping']
            question = values['question']
            pred_text = preds['desc']
            pred_boxes = preds['box']
            gt_text = gt['desc']
            gt_boxes = gt['box']
            pred_boxes_cp = pred_boxes.copy()

            if video_id == "7MRKY":
                pred_boxes = box_cxcywh_to_xyxy(torch.tensor(pred_boxes))
                pred_boxes = PostProcess()(pred_boxes, vid_orig_size.repeat(pred_boxes.shape[0], 1))  # 32x10x4

                # gt_boxes = box_cxcywh_to_xyxy(torch.tensor(gt_boxes))
                # gt_boxes = PostProcess()(gt_boxes, vid_orig_size.repeat(gt_boxes.shape[0], 1))  # 32x10x4

                if pred_boxes.ndim == 3:
                    pred_boxes = pred_boxes[:, 0, :]

                total_boxes = np.stack((np.array(gt_boxes), pred_boxes.numpy()), axis=1)

                video_save_path = os.path.join(
                    args.save_dir,
                    video_id,
                    q_id)
                if not os.path.exists(video_save_path):
                    os.makedirs(video_save_path)
                # extract actual images from the video to process them adding boxes
                draw_and_save_rects(osp.join(args.video_dir, video_id), video_frame_ids, total_boxes, video_save_path) # TODO draw gt as well

                # Add question/answer/preds
                file_path = osp.join(video_save_path, 'pred.txt')

                # Open the file in write mode ('w')
                with open(file_path, 'w') as file:
                    # Write some text to the file
                    file.write(question+" "+gt_text+ "\n")
                    file.write("pred: "+pred_text + "\n")

                print(f"Video saved in {video_save_path}")
                for i, box in enumerate(pred_boxes):
                    print("pred_box orig", pred_boxes_cp[i])
                    print("pred_box scaled",box)
                    print("gt box scaled",gt_boxes[i])
                break



if __name__ == "__main__":
    # set random seeds
    args = get_args()
    torch.backends.cudnn.enabled = False
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
        
    main(args)
