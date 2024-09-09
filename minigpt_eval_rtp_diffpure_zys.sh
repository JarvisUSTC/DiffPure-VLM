output_path=$1
image_prompt_path=$2
dp_f_step=$3
dp_r_step=$4

# if output path does not exist, create it
if [ ! -d $output_path ]; then
    mkdir -p $output_path
fi

# for i in {9..15}
# do
#   j=$((i-9))
#   # 限制可见GPU
#   bash -c "CUDA_VISIBLE_DEVICES=10 python minigpt_inference.py --cfg-path eval_configs/minigpt4_eval.yaml --gpu-id 0 --image_file $image_prompt_path --input_file harmful_corpus/rtp_prompts_challenges_$j.jsonl --output_file $output_path/rtp_prompts_challenges_$j.jsonl" > $j.log 2>&1 &
# done

CUDA_VISIBLE_DEVICES=10 python minigpt_inference_diffpure.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0 --image_file $image_prompt_path --input_file harmful_corpus/rtp_prompts_challenges_0.jsonl --output_file $output_path/rtp_prompts_challenges_0.jsonl --def_max_timesteps $dp_f_step --def_num_denoising_steps $dp_r_step --def_sampling_method ddpm

echo "part 1 done"

CUDA_VISIBLE_DEVICES=10 python minigpt_inference_diffpure.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0 --image_file $image_prompt_path --input_file harmful_corpus/rtp_prompts_challenges_1.jsonl --output_file $output_path/rtp_prompts_challenges_1.jsonl --def_max_timesteps $dp_f_step --def_num_denoising_steps $dp_r_step --def_sampling_method ddpm

echo "part 2 done"

CUDA_VISIBLE_DEVICES=10 python minigpt_inference_diffpure.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0 --image_file $image_prompt_path --input_file harmful_corpus/rtp_prompts_challenges_2.jsonl --output_file $output_path/rtp_prompts_challenges_2.jsonl --def_max_timesteps $dp_f_step --def_num_denoising_steps $dp_r_step --def_sampling_method ddpm

echo "part 3 done"

CUDA_VISIBLE_DEVICES=10 python minigpt_inference_diffpure.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0 --image_file $image_prompt_path --input_file harmful_corpus/rtp_prompts_challenges_3.jsonl --output_file $output_path/rtp_prompts_challenges_3.jsonl --def_max_timesteps $dp_f_step --def_num_denoising_steps $dp_r_step --def_sampling_method ddpm

echo "part 4 done"

CUDA_VISIBLE_DEVICES=10 python minigpt_inference_diffpure.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0 --image_file $image_prompt_path --input_file harmful_corpus/rtp_prompts_challenges_4.jsonl --output_file $output_path/rtp_prompts_challenges_4.jsonl --def_max_timesteps $dp_f_step --def_num_denoising_steps $dp_r_step --def_sampling_method ddpm

echo "part 5 done"

CUDA_VISIBLE_DEVICES=10 python minigpt_inference_diffpure.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0 --image_file $image_prompt_path --input_file harmful_corpus/rtp_prompts_challenges_5.jsonl --output_file $output_path/rtp_prompts_challenges_5.jsonl --def_max_timesteps $dp_f_step --def_num_denoising_steps $dp_r_step --def_sampling_method ddpm

echo "part 6 done"

CUDA_VISIBLE_DEVICES=10 python minigpt_inference_diffpure.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0 --image_file $image_prompt_path --input_file harmful_corpus/rtp_prompts_challenges_6.jsonl --output_file $output_path/rtp_prompts_challenges_6.jsonl --def_max_timesteps $dp_f_step --def_num_denoising_steps $dp_r_step --def_sampling_method ddpm

echo "part 7 done"

CUDA_VISIBLE_DEVICES=10 python minigpt_inference_diffpure.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0 --image_file $image_prompt_path --input_file harmful_corpus/rtp_prompts_challenges_7.jsonl --output_file $output_path/rtp_prompts_challenges_7.jsonl --def_max_timesteps $dp_f_step --def_num_denoising_steps $dp_r_step --def_sampling_method ddpm

echo "part 8 done"

# echo "Started all processes"

# # wait for all processes to finish
# wait

# echo "All processes finished"

## merge all the jsonl outputs
# loop
for i in {0..7}
do
    cat $output_path/rtp_prompts_challenges_$i.jsonl >> $output_path/rtp_prompts_challenges.jsonl
done

echo "Merged all jsonl outputs"

# remove the individual jsonl outputs
for i in {0..7}
do
    rm $output_path/rtp_prompts_challenges_$i.jsonl
done

echo "Removed individual jsonl outputs"

# # move the logs to output path
# for i in {0..7}
# do
#     mv $i.log $output_path
# done

# echo "Moved logs"

# evaluate the generated prompts
CUDA_VISIBLE_DEVICES=15 python get_metric.py --input $output_path/rtp_prompts_challenges.jsonl --output $output_path/rtp_prompts_challenges_metrics.jsonl

echo "Evaluated the generated prompts"

# cal metric
python cal_metrics_markdown.py --input $output_path/rtp_prompts_challenges_metrics.jsonl

echo "Calculated metrics"