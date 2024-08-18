# On the Adversarial Robustness of DiffPure in VLM

## Plan

- [x] Baseline Experments I: Reproduced the attack success rate in an undefended scenario.
- [x] Baseline Experments II: Reproduced the attack success rate in DiffPure scenario.
- [x] Advanced Experiments I: Leverage the REDiffPure to test the robustness of DiffPure.

## Environment Setup

```bash
conda env create -f environment.yml
conda activate minigpt4
```

## Pretrained Model Preparation

```bash
mkdir -p ckpts/
ln -s /blob_msra/zhuzho_container/v-jiaweiwang/LLMs/vicuna ckpts/vicuna
ln -s /blob_msra/zhuzho_container/v-jiaweiwang/pretrained_models/pretrained_minigpt4.pth ckpts/pretrained_minigpt4.pth
mkdir -p ckpts/diffpure_models/diffusion/Guide_Diffusion/
ln -s /blob_msra/zhuzho_container/v-jiaweiwang/pretrained_models/256x256_diffusion_uncond.pt ckpts/diffpure_models/diffusion/Guide_Diffusion/256x256_diffusion_uncond.pt
```
Minigpt4: https://drive.google.com/file/d/1a4zLvaiDBr-36pasffmgpvH5P7CKmpze/view

Vicuna: https://huggingface.co/Vision-CAIR/vicuna/tree/main

Diffusion Model: https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt

## How to Run

To train with DiffPure, you can run the following command:

```bash
python minigpt_visual_attack_diffpure.py --cfg_path eval_configs/minigpt4_eval.yaml --gpu_id 0 --n_iters 5000 --constrained --eps 16 --alpha 1 --save_dir outputs/visual_constrained_eps_16_diffpure_30_1_ddpm --att_max_timesteps 30 --att_num_denoising_steps 1 --att_sampling_method ddpm --eot 1
```
Please refer to "adversarial_images/prompt_constrained_16_diff_30_1_ddpm.bmp" for the optimized adversarial image based on the above command.

To evaluate the robustness of DiffPure, you can run the following command:
- Request and place your [Perspective API](https://perspectiveapi.com/) key in `.perspective_api_key`.
```bash
bash minigpt_eval_rtp_diffpure.sh {output_path} {image_path}
```
Please replace the `{output_path}` with the path to the save dir and `{image_path}` with the path to the adversarial image.

Prevous results are saved in "results/"