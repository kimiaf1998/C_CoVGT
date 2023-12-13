GPU=$1
CHECKPOINT="../data/save_models/STAR/CoVGT/best_model.pth"
OUTPUT_PATH="../data/save_models/STAR/CoVGT/"

CUDA_VISIBLE_DEVICES=$GPU python demo_star.py --load=$CHECKPOINT \
--dataset=STAR \
--qmax_words=30 \
--amax_words=38 \
--video_max_len=32 \
--vid_id "N5PLR" \
--qid "Interaction_T1_104" \
--question "Which object was taken by the person?" \
--answer "The clothes." \
--choices "The clothes." "The cup/glass/bottle." "The sandwich." "he book." \
--save_dir $OUTPUT_PATH \
--device "cpu"