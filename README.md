# On the Adversarial Robustness of DiffPure in VLM

## Plan

- [x] Baseline Experments I: Reproduced the attack success rate in an undefended scenario.
- [x] Baseline Experments II: Reproduced the attack success rate in DiffPure scenario.
- [x] Advanced Experiments I: Leverage the REDiffPure to test the robustness of DiffPure.
- [x] Baseline Experments III: Train adversial images under Instruct BLIP.
- [x] Baseline Experments IV: Train adversial images under LLaVA-LLaMA-2.

## TODO

Please run test scripts in `ExpPlan_MiniGPT4.md` for more details.


## Environment Setup

Based on my experiment, I suggest to create 3 environments for 3 VLMs.

```bash
# For MiniGPT-4:
conda env create -f environment.yml
conda activate minigpt4
# For Instruct Blip:
conda create --name IB python=3.9
conda activate IB
cd LAVIS
pip install -e .
# For LLaVA-LLaMA-2:
conda create --name LL2 python=3.9
conda activate LL2
cd LLaVA
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

## Pretrained Model Preparation

```bash
mkdir -p ckpts/
ln -s /blob_msra/zhuzho_container/v-jiaweiwang/LLMs/vicuna ckpts/vicuna
ln -s /blob_msra/zhuzho_container/v-jiaweiwang/pretrained_models/pretrained_minigpt4.pth ckpts/pretrained_minigpt4.pth
mkdir -p ckpts/diffpure_models/diffusion/Guide_Diffusion/
ln -s /blob_msra/zhuzho_container/v-jiaweiwang/pretrained_models/256x256_diffusion_uncond.pt ckpts/diffpure_models/diffusion/Guide_Diffusion/256x256_diffusion_uncond.pt
```
Above commands are used in MSRA environment.

For other users, download the corresponding pretrained models and save it in 'ckpts' folder.

Note: Before downloading pretrained models, please ensures that you have >100GB free space in your machine.

### Universal

BERT Tokenizer: https://huggingface.co/google-bert/bert-base-uncased

(Recommend) Download all files using `git lfs` and save the folder to `ckpts` folder.

### For minigpt-4:

Minigpt4: https://drive.google.com/file/d/1a4zLvaiDBr-36pasffmgpvH5P7CKmpze/view

Vicuna: https://huggingface.co/Vision-CAIR/vicuna/tree/main

Download all files and save in `./ckpts/vicuna-13b-v1.1`

Diffusion Model: https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt

Q_former: https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth

EVA_VIT_G: https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth

### For Instruct BLIP:

Vicuna-13B-v1.1: https://huggingface.co/lmsys/vicuna-13b-v1.1

Download all files and save in `./ckpts/vicuna-13b-v1.1`

instruct_blip_vicuna13b_trimmed.pth: https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/InstructBLIP/instruct_blip_vicuna13b_trimmed.pth

### For LLaVA-LLaMA-2

llava-llama-2-13b-chat: https://huggingface.co/liuhaotian/llava-llama-2-13b-chat-lightning-preview

Download all files and save in `./ckpts/llava_llama_2_13b_chat_freeze`

### File structure in `./ckpts` folder

```bash
bert-base-uncased
    -- files from huggingface
diffpure_models
    -- classifier
        -- ResNet50
    -- diffusion
        -- Guide_Diffusion
            256x256_diffusion_uncond.pt
        -- Score_SDE
llava_llama_2_13b_chat_freeze
    -- files from huggingface
victuna
    -- files from huggingface
victuna-13b-v1.1
    -- files from huggingface
blip2_pretrained_flant5xxl.pth
eva_vit_g.pth
instruct_blip_vicuna13b_trimmed.pth
pretrained_minigpt4.pth
```

## How to Run

### Instruct BLIP

To train with Instruct BLIP, run following commands:

* Attack

  ```bash
  python -u instructblip_visual_attack.py --n_iters 5000 --constrained --save_dir results_blip_constrained_16 --eps 16 --alpha 1
  python -u instructblip_visual_attack.py --n_iters 5000 --constrained --save_dir results_blip_constrained_32 --eps 32 --alpha 1
  python -u instructblip_visual_attack.py --n_iters 5000 --constrained --save_dir results_blip_constrained_64 --eps 64 --alpha 1
  python -u instructblip_visual_attack.py --n_iters 5000 --save_dir results_blip_unconstrained --alpha 1
  ```
* Testing on the RealToxicityPrompts Dataset

  Make inference on the dataset:

  ```bash
  python instructblip_inference.py --image_file path_to_the_adversarial_example --output_file result.jsonl
  ```

  The `get_metric.py` will calculate the toxic scores using both Perspective API and [Detoxify](https://github.com/unitaryai/detoxify).

  ```bash
  python get_metric.py --input result.jsonl --output result_eval.jsonl
  ```

  Then, you can run `cal_metrics.py` to summarize the evaluation results from the two evaluation:

  ```bash
  python cal_metrics.py --input result_eval.jsonl
  ```

### LLaVA-LLaMA-2

* Attack

  ```bash
  python -u llava_llama_v2_visual_attack.py --n_iters 5000 --constrained --save_dir results_llava_llama_v2_constrained_16 --eps 16 --alpha 1
  python -u llava_llama_v2_visual_attack.py --n_iters 5000 --constrained --save_dir results_llava_llama_v2_constrained_32 --eps 32 --alpha 1
  python -u llava_llama_v2_visual_attack.py --n_iters 5000 --constrained --save_dir results_llava_llama_v2_constrained_64 --eps 64 --alpha 1
  python -u llava_llama_v2_visual_attack.py --n_iters 5000 --save_dir results_llava_llama_v2_unconstrained --alpha 1
  ```

* Testing on the RealToxicityPrompts Dataset

  Make inference on the dataset:

  ```bash
  python -u llava_llama_v2_inference.py --image_file path_to_the_adversarial_example --output_file result.jsonl
  ```

  The `get_metric.py` will calculate the toxic scores using both Perspective API and [Detoxify](https://github.com/unitaryai/detoxify).

  ```bash
  python get_metric.py --input result.jsonl --output result_eval.jsonl
  ```

  Then, you can run `cal_metrics.py` to summarize the evaluation results from the two evaluation:

  ```bash
  python cal_metrics.py --input result_eval.jsonl
  ```

### MiniGPT-4 with DiffPure

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

Prevous results are saved in `results/`