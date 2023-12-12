from models.tubedetr import build
from models.CoVGT import VGT

def build_model(args):
    if args.lan == 'BERT':
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    elif args.lan == 'RoBERTa':
        from transformers import RobertaTokenizerFast, RobertaTokenizer
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    # Space-time decoder

    tube_detector, loc_criterion = build(args)
    loc_criterion.cuda()

    model = VGT(
        tokenizer = tokenizer,
        tube_detector=tube_detector,
        feature_dim=args.feature_dim,
        word_dim=args.word_dim,
        N=args.n_layers,
        d_model=args.embd_dim,
        d_ff=args.ff_dim,
        h=args.n_heads,
        dropout=args.dropout,
        T=args.video_max_len,
        Q=args.qmax_words,
        vocab_size = tokenizer.vocab_size,
        baseline=args.baseline,
        bnum=args.bnum,
        lan=args.lan)

    return model, tokenizer
