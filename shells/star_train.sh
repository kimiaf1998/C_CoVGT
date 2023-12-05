GPU=$1
CUDA_VISIBLE_DEVICES=$GPU python main.py --checkpoint_dir=STAR \
	--dataset=STAR \
	--mc=4 \
	--bnum=10 \
	--epochs=20 \
	--lr=0.00001 \
	--qmax_words=30 \
	--amax_words=38 \
	--video_max_len=32 \
	--batch_size=64 \
	--batch_size_val=64 \
	--num_thread_reader=8 \
	--mlm_prob=0 \
	--cl_loss=1 \
	--n_layers=1 \
	--embd_dim=512 \
	--ff_dim=1024 \
	--dropout=0.3 \
	--seed=666 \
	--lan="RoBERTa" \
	--save_dir='../data/save_models/STAR/CoVGT/' \
	--pretrain_path=../data/save_models/webvid180K/co_e1.pth \
	--num_queries=10 \
	--ema \

	
	
	
