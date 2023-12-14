GPU=$1
CHECKPOINT="../data/save_models/STAR/CoVGT/Final_Res/best_model_mviou_13.79_acc_44.90.pth"
OUTPUT_PATH="../data/save_models/STAR/CoVGT/Final_Res/Visualization"

CUDA_VISIBLE_DEVICES=$GPU python demo_star.py --load=$CHECKPOINT \
--dataset=STAR \
--qmax_words=30 \
--amax_words=38 \
--video_max_len=32 \
--vid_id "5T0NX" \
--qid "Prediction_T1_45" \
--question "What will the person do next?" \
--answer "Take the dish." \
--choices "Take the dish." "Eat the medicine." "Put down the clothes." "Take the paper/notebook." \
--save_dir $OUTPUT_PATH \
--device "cpu"