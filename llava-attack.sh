GPU_ID=$1

MODEL_PATH='/home/t-jiaweiwang/Project/LLaVA/checkpoints/llava-v1.5-7b-finetune_RobustVLGuard_backbone_lora_4k_2k_4e_STD005-015_P07_r16_4e-5/checkpoint-3000'

if [ ! -d "$MODEL_PATH/attack" ]; then
  mkdir -p "$MODEL_PATH/attack"
fi

CUDA_VISIBLE_DEVICES=$GPU_ID python llava_v1_5_visual_attack.py \
--model-path $MODEL_PATH \
--eps 32 \
--alpha 1 \
--n_iters 5000 \
--constrained \
--save_dir $MODEL_PATH/attack 2>&1 | tee -a "$MODEL_PATH/attack/training_log.txt"

wait

save_image_path="$MODEL_PATH/attack/LLaVA-v15-7B-eps32-alpha1-iters5000-constrained.bmp"

bash omi_eval_rtp.sh $MODEL_PATH/attack/LLaVA-v15-7B-eps32-alpha1-iters5000-constrained/attack/ $save_image_path $MODEL_PATH &
bash omi_eval_rtp_diffpure.sh $MODEL_PATH/attack/LLaVA-v15-7B-eps32-alpha1-iters5000-constrained/diffpure_50_50/ $save_image_path $MODEL_PATH 50 &

wait
bash omi_eval_rtp_diffpure.sh $MODEL_PATH/attack/LLaVA-v15-7B-eps32-alpha1-iters5000-constrained/diffpure_100_100/ $save_image_path $MODEL_PATH 100 &
bash omi_eval_rtp_diffpure.sh $MODEL_PATH/attack/LLaVA-v15-7B-eps32-alpha1-iters5000-constrained/diffpure_150_150/ $save_image_path $MODEL_PATH 150 &

wait
