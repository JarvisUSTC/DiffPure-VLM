output_folder=$1
image_prompt_path=$2
constraint=$3
dp_f_step=$4
dp_r_step=$5
gpu_device=$6

# if output path does not exist, create it
if [ ! -d $output_folder ]; then
    mkdir -p $output_folder
fi

# for i in {9..15}
# do
#   j=$((i-9))
#   # 限制可见GPU
#   bash -c "CUDA_VISIBLE_DEVICES=10 python minigpt_inference.py --cfg-path eval_configs/minigpt4_eval.yaml --gpu-id 0 --image_file $image_prompt_path --input_file harmful_corpus/rtp_prompts_challenges_$j.jsonl --output_file $output_path/rtp_prompts_challenges_$j.jsonl" > $j.log 2>&1 &
# done

CUDA_VISIBLE_DEVICES=$gpu_device python diff_explore.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0 --output_folder $output_folder --image_file $image_prompt_path --clean_image ./adversarial_images/clean.jpeg --constraint $constraint --def_max_timesteps $dp_f_step --def_num_denoising_steps $dp_r_step --def_sampling_method ddpm
