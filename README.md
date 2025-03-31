# 🚀 Robust-VLGuard & DiffPure-VLM: Safeguarding Vision-Language Models

Welcome! This repository hosts the official implementation of our paper, **"Safeguarding Vision-Language Models: Mitigating Vulnerabilities to Gaussian Noise in Perturbation-based Attacks"**.

---

## 🌟 What’s New?

We provide cutting-edge solutions for enhancing the robustness of Vision-Language Models (VLMs) against Gaussian noise and adversarial attacks. Specifically:

- 🎯 **Robust-VLGuard**: A pioneering multimodal safety dataset addressing both aligned and misaligned image-text pair scenarios.

![RobustVLGuard](assets/Robust-VLGuard.png)

- 🛡️ **DiffPure-VLM**: A novel defense framework leveraging the power of diffusion models to effectively neutralize adversarial noise by transforming it into Gaussian-like noise, significantly boosting VLM resilience.

![DiffPure-VLM](assets/DiffPure-VLM.png)

---

## ✨ Key Contributions

- 🔍 Conducted a comprehensive vulnerability analysis exposing mainstream VLM weaknesses against Gaussian noise.
- 📚 Created Robust-VLGuard, designed specifically to bolster model robustness without compromising helpfulness and safety alignment.
- ⚙️ Proposed DiffPure-VLM, an innovative defense pipeline that effectively counters complex optimization-based adversarial perturbation attacks.
- 📈 Demonstrated superior performance of our approach compared to existing baseline methods through rigorous benchmarking.

---


## ⚡ Quickstart

### 🛠️ Installation

Different models require different environments. We provide a `conda` environment file for each model in env_configs. For example, to set up the environment for most VLMs, run:

```bash
conda env create -f env_configs/environment-omi.yml
conda activate Omi-Environment
```

### 🛠️ Pretrained Model Preparation

```bash
mkdir -p ckpts/
ln -s your_path/vicuna ckpts/vicuna
ln -s your_path/pretrained_minigpt4.pth ckpts/pretrained_minigpt4.pth
mkdir -p ckpts/diffpure_models/diffusion/Guide_Diffusion/
ln -s your_path/256x256_diffusion_uncond.pt ckpts/diffpure_models/diffusion/Guide_Diffusion/256x256_diffusion_uncond.pt
```
Minigpt4: https://drive.google.com/file/d/1a4zLvaiDBr-36pasffmgpvH5P7CKmpze/view

Vicuna: https://huggingface.co/Vision-CAIR/vicuna/tree/main

Diffusion Model: https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt

### 📥 Dataset Setup

RealToxicityPrompts Benchmark has been downloaded and preprocessed. You can find the dataset in the `harmful_corpus/` directory.

You can download the Robust-VLGuard dataset from [Google Drive](https://drive.google.com/file/d/1EoNz3Nzh3sq8ZxOlZC2oB603OVSpRJoO/view?usp=sharing).

We also provide a noisy version of MMVet benchmark, which can be downloaded from [Google Drive](https://drive.google.com/file/d/1mmkDKH0BlgpJwJkxFklXp7EtSa1BE6XB/view?usp=sharing).

### 🚀 Fine-tuning VLMs

Our Robust-VLGuard dataset has been preprocessed and is ready for fine-tuning. You can use the official code from the respective VLMs to fine-tune them with our dataset. For example, to fine-tune llava, refer to the finetuning_scripts/llava.sh script.

### Evaluation on RealToxicityPrompts

To evaluate the VLMs on the RealToxicityPrompts benchmark, you can use the following command:

```bash
bash general_scripts/omi_eval_rtp.sh {OUTPUT_PATH} adversarial_images/clean.jpeg {MODEL_PATH}
```

or 

```bash
bash general_scripts/omi_eval_rtp.sh {OUTPUT_PATH} adversarial_images_add_noise_G30/clean.jpeg {MODEL_PATH}
```

adversarial_images/clean.jpeg refers to the clean image, and adversarial_images_add_noise_G30/clean.jpeg refers to the image with Gaussian noise added. You can easily change the image path in the script.

### Evaluation on MMVet

To evaluate LLaVA on the MMVet benchmark, you can use the following command:

```bash
python llava_inference_mmvet.py --model_path {MODEL_PATH} --clean --output_path {OUTPUT_PATH}
```

or 

```bash
python llava_inference_mmvet.py --model_path {MODEL_PATH} --output_path {OUTPUT_PATH}
```

--clean refers to using the clean images.

For MiniGPT4, you can refer to minigpt_inference_mmvet.py.

### Optimized-based Adversarial Attack
To generate adversarial examples for LLaVA, you can use the following command:

```bash
bas llava-attack.sh {GPU_ID} {OUTPUT_PATH} {MODEL_PATH} {EPSILON}
```

EPSILON is the perturbation strength, and it can be set to 16, 32, or 64.

For MiniGPT4, you can refer to the minigpt_visual_attack.py.

### 🔐 Deploying DiffPure-VLM Defense

For most VLMs, you can run the following command to deploy the DiffPure-VLM defense pipeline:

```bash
bash general_scripts/omi_eval_rtp_diffpure.sh {output_path} {image_prompt_path} {model_path} {def_num_denoising_steps}
```

For minigpt4 and qwen25_vl, you can refer to minigpt_scripts/minigpt_eval_rtp_diffpure_single_gpu.sh and qwen25_vl_scripts/qwen25_vl_rtp_diffpure.sh respectively.

---

## 📊 Experimental Results
Explore comprehensive experimental results and detailed analysis in our paper and supplementary materials. Specific results can be found in results/.

![Table 1](assets/Table2.png)
![Table 2](assets/Table4.png)
![Table 3](assets/Table5.png)


---

## 📜 Citation
Please cite our paper if you find this work useful:


---

## 📃 License
Distributed under the MIT License. See `LICENSE` for more details.

---

## 📫 Contact
For inquiries or discussions, please reach out to [jarvisustc@gmail.com]. We welcome collaboration and feedback!