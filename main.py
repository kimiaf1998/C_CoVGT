import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import os.path as osp
import logging

from transformers import get_cosine_schedule_with_warmup
from args import get_args
from models.CoVGT import VGT
import models.tubedetr as tube_detector
from models.Tube_CoVGT import build_model
from loss import LogSoftmax
from models.space_time_decoder import build_transformer
from models.tubedetr import TubeDecoder, build
from util import compute_a2v, load_model_by_key, save_to
from dataloader.cvqa_loader import get_videoqa_loaders
from train.train_covgt import train, eval
from tqdm import tqdm


def main(args):
    if not (os.path.isdir(args.save_dir)):
        os.mkdir(os.path.join(args.save_dir))
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s"
    )
    logFormatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    rootLogger = logging.getLogger()
    fileHandler = logging.FileHandler(os.path.join(args.save_dir, "stdout.log"), "w+")
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    logging.info(args)

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
        "loss_vqa": 1,
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
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 0, len(train_loader) * args.epochs
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
        for epoch in tqdm(range(args.epochs), desc="Training on epoch", unit="epoch"):
            train(model, train_loader, a2v, optimizer, qa_criterion, loc_criterion, weight_dict, scheduler, epoch, args, tokenizer)
            outputs = eval(model, val_loader, a2v, args, test=False, tokenizer=tokenizer)
            val_iou = outputs["metrics"]["m_viou"]
            val_acc = outputs["metrics"]["acc"]
            results = outputs["results"]
            if val_iou > best_val_viou:
                best_val_viou = val_iou
                best_epoch = epoch
                torch.save(
                    model.state_dict(), os.path.join(args.save_dir, f'best_model_mviou_{val_iou}.pth')
                )
                logging.info(f"Best models have been saved in {os.path.join(args.save_dir, f'best_model_mviou_{val_iou}.pth')}")
                print(f"Best models have been saved in {os.path.join(args.save_dir, f'best_model_mviou_{val_iou}.pth')}")
                save_path = osp.join(args.save_dir, 'val-res.json')
                save_to(save_path, results)
            if args.dataset == 'webvid': 
                ep_file = os.path.join(args.save_dir, f"e{epoch}.pth")
                torch.save(model.state_dict(), ep_file)
                logging.info('Save to '+ep_file)
        logging.info(f"Best val models at epoch {best_epoch + 1} with m_viou {best_val_viou:.2f} and acc {val_acc:.2f}")
        print(f"Best val models at epoch {best_epoch + 1} with m_viou {best_val_viou:.2f} and acc {val_acc:.2f}")
    else:
        # Evaluate on val (=test) set
        outputs = eval(model, test_loader, a2v, args, test=False, tokenizer=tokenizer)  # zero-shot VideoQA
        results = outputs["results"]
        save_path = osp.join(args.save_dir, 'val-res0.json')
        save_to(save_path, results)
        print("Model Validation Results:")
        logging.info("Model Validation Results:")
        step = "Test"
        for k, v in outputs["metrics"].items():
            print(f"{step} {k}: {v:.2%}")
            logging.info(f"{step} {k}: {v:.2%}")


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
