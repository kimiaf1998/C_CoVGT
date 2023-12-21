GPU=$1
CHECKPOINT="../data/save_models/STAR/CoVGT/Final_Res/best_model_mviou_13.79_acc_44.90.pth"

CUDA_VISIBLE_DEVICES=$GPU python main.py --checkpoint_dir=STAR \
	--dataset=STAR \
	--mc=4 \
	--bnum=10 \
	--epochs=35 \
	--lr=5e-5 \
	--qmax_words=30 \
	--amax_words=38 \
	--video_max_len=32 \
	--batch_size=64 \
	--batch_size_val=64 \
	--num_thread_reader=8 \
	--mlm_prob=0 \
	--cl_loss=2 \
	--n_layers=4 \
	--embd_dim=512 \
	--ff_dim=1024 \
	--dropout=0.3 \
	--seed=666 \
	--lan="RoBERTa" \
	--save_dir='../data/save_models/STAR/CoVGT/Last_Res_12_15_11am' \
	--pretrain_path=../data/save_models/nextqa/CoVGT_FTCoWV/best_model.pth\
	--num_queries=10 \
	--no_sted \

	
	
	
