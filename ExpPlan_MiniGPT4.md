# Experiment plan

## DiffPure step sensitivity in different constraints

Currently, we have experiment results in 30 steps.

Now we go further.

In the test script, the template is 

`bash minigpt_eval_rtp_diffpure_zys.sh {output_dir} {image_path} {ddpm_forward_step} {ddpm_reverse_step} {gpu_device_id}`


## MiniGPT-4

### 50 steps

#### RAW adversiral images
```bash
bash minigpt_eval_rtp_diffpure_zys.sh ./results/constrained_16_ddpm_50_50 ./adversarial_images/prompt_constrained_16.bmp 50 50 3
bash minigpt_eval_rtp_diffpure_zys.sh ./results/constrained_32_ddpm_50_50 ./adversarial_images/prompt_constrained_32.bmp 50 50 4
bash minigpt_eval_rtp_diffpure_zys.sh ./results/constrained_64_ddpm_50_50 ./adversarial_images/prompt_constrained_64.bmp 50 50 5
bash minigpt_eval_rtp_diffpure_zys.sh ./results/unconstrained_ddpm_50_50 ./adversarial_images/prompt_unconstrained.bmp 50 50 6
```

#### RAW adversiral images with gaussian noise (sigma=30)
```bash
bash minigpt_eval_rtp_diffpure_zys.sh ./results/clean_G30_ddpm_50_50 ./adversarial_images_add_noise_G30/clean.jpeg 50 50 3
bash minigpt_eval_rtp_diffpure_zys.sh ./results/constrained_16_G30_ddpm_50_50 ./adversarial_images_add_noise_G30/prompt_constrained_16.bmp 50 50 4
bash minigpt_eval_rtp_diffpure_zys.sh ./results/constrained_32_G30_ddpm_50_50 ./adversarial_images_add_noise_G30/prompt_constrained_32.bmp 50 50 5
bash minigpt_eval_rtp_diffpure_zys.sh ./results/constrained_64_G30_ddpm_50_50 ./adversarial_images_add_noise_G30/prompt_constrained_64.bmp 50 50 6
bash minigpt_eval_rtp_diffpure_zys.sh ./results/unconstrained_G30_ddpm_50_50 ./adversarial_images_add_noise_G30/prompt_unconstrained.bmp 50 50 7
```

#### RAW adversiral images with gaussian noise (sigma=50)
```bash
bash minigpt_eval_rtp_diffpure_zys.sh ./results/clean_G50_ddpm_50_50 ./adversarial_images_add_noise_G50/clean.jpeg 50 50 3
bash minigpt_eval_rtp_diffpure_zys.sh ./results/constrained_16_G50_ddpm_50_50 ./adversarial_images_add_noise_G50/prompt_constrained_16.bmp 50 50 4
bash minigpt_eval_rtp_diffpure_zys.sh ./results/constrained_32_G50_ddpm_50_50 ./adversarial_images_add_noise_G50/prompt_constrained_32.bmp 50 50 5
bash minigpt_eval_rtp_diffpure_zys.sh ./results/constrained_64_G50_ddpm_50_50 ./adversarial_images_add_noise_G50/prompt_constrained_64.bmp 50 50 6
bash minigpt_eval_rtp_diffpure_zys.sh ./results/unconstrained_G50_ddpm_50_50 ./adversarial_images_add_noise_G50/prompt_unconstrained.bmp 50 50 7
```

#### RAW adversiral images with gaussian noise (sigma=75)
```bash
bash minigpt_eval_rtp_diffpure_zys.sh ./results/clean_G75_ddpm_50_50 ./adversarial_images_add_noise_G75/clean.jpeg 50 50 3
bash minigpt_eval_rtp_diffpure_zys.sh ./results/constrained_16_G75_ddpm_50_50 ./adversarial_images_add_noise_G75/prompt_constrained_16.bmp 50 50 4
bash minigpt_eval_rtp_diffpure_zys.sh ./results/constrained_32_G75_ddpm_50_50 ./adversarial_images_add_noise_G75/prompt_constrained_32.bmp 50 50 5
bash minigpt_eval_rtp_diffpure_zys.sh ./results/constrained_64_G75_ddpm_50_50 ./adversarial_images_add_noise_G75/prompt_constrained_64.bmp 50 50 6
bash minigpt_eval_rtp_diffpure_zys.sh ./results/unconstrained_G75_ddpm_50_50 ./adversarial_images_add_noise_G75/prompt_unconstrained.bmp 50 50 7
```


### 100 steps

#### RAW adversiral images
```bash
bash minigpt_eval_rtp_diffpure_zys.sh ./results/constrained_16_ddpm_100_100 ./adversarial_images/prompt_constrained_16.bmp 100 100 3
bash minigpt_eval_rtp_diffpure_zys.sh ./results/constrained_32_ddpm_100_100 ./adversarial_images/prompt_constrained_32.bmp 100 100 4
bash minigpt_eval_rtp_diffpure_zys.sh ./results/constrained_64_ddpm_100_100 ./adversarial_images/prompt_constrained_64.bmp 100 100 5
bash minigpt_eval_rtp_diffpure_zys.sh ./results/unconstrained_ddpm_100_100 ./adversarial_images/prompt_unconstrained.bmp 100 100 6
```

#### RAW adversiral images with gaussian noise (sigma=30)
```bash
bash minigpt_eval_rtp_diffpure_zys.sh ./results/clean_G30_ddpm_100_100 ./adversarial_images_add_noise_G30/clean.jpeg 100 100 3
bash minigpt_eval_rtp_diffpure_zys.sh ./results/constrained_16_G30_ddpm_100_100 ./adversarial_images_add_noise_G30/prompt_constrained_16.bmp 100 100 4
bash minigpt_eval_rtp_diffpure_zys.sh ./results/constrained_32_G30_ddpm_100_100 ./adversarial_images_add_noise_G30/prompt_constrained_32.bmp 100 100 5
bash minigpt_eval_rtp_diffpure_zys.sh ./results/constrained_64_G30_ddpm_100_100 ./adversarial_images_add_noise_G30/prompt_constrained_64.bmp 100 100 6
bash minigpt_eval_rtp_diffpure_zys.sh ./results/unconstrained_G30_ddpm_100_100 ./adversarial_images_add_noise_G30/prompt_unconstrained.bmp 100 100 7
```

#### RAW adversiral images with gaussian noise (sigma=50)
```bash
bash minigpt_eval_rtp_diffpure_zys.sh ./results/clean_G50_ddpm_100_100 ./adversarial_images_add_noise_G50/clean.jpeg 100 100 3
bash minigpt_eval_rtp_diffpure_zys.sh ./results/constrained_16_G50_ddpm_100_100 ./adversarial_images_add_noise_G50/prompt_constrained_16.bmp 100 100 4
bash minigpt_eval_rtp_diffpure_zys.sh ./results/constrained_32_G50_ddpm_100_100 ./adversarial_images_add_noise_G50/prompt_constrained_32.bmp 100 100 5
bash minigpt_eval_rtp_diffpure_zys.sh ./results/constrained_64_G50_ddpm_100_100 ./adversarial_images_add_noise_G50/prompt_constrained_64.bmp 100 100 6
bash minigpt_eval_rtp_diffpure_zys.sh ./results/unconstrained_G50_ddpm_100_100 ./adversarial_images_add_noise_G50/prompt_unconstrained.bmp 100 100 7
```

#### RAW adversiral images with gaussian noise (sigma=75)
```bash
bash minigpt_eval_rtp_diffpure_zys.sh ./results/clean_G75_ddpm_100_100 ./adversarial_images_add_noise_G75/clean.jpeg 100 100 3
bash minigpt_eval_rtp_diffpure_zys.sh ./results/constrained_16_G75_ddpm_100_100 ./adversarial_images_add_noise_G75/prompt_constrained_16.bmp 100 100 4
bash minigpt_eval_rtp_diffpure_zys.sh ./results/constrained_32_G75_ddpm_100_100 ./adversarial_images_add_noise_G75/prompt_constrained_32.bmp 100 100 5
bash minigpt_eval_rtp_diffpure_zys.sh ./results/constrained_64_G75_ddpm_100_100 ./adversarial_images_add_noise_G75/prompt_constrained_64.bmp 100 100 6
bash minigpt_eval_rtp_diffpure_zys.sh ./results/unconstrained_G75_ddpm_100_100 ./adversarial_images_add_noise_G75/prompt_unconstrained.bmp 100 100 7
```


### 150 steps

#### RAW adversiral images
```bash
bash minigpt_eval_rtp_diffpure_zys.sh ./results/constrained_16_ddpm_150_150 ./adversarial_images/prompt_constrained_16.bmp 150 150 3
bash minigpt_eval_rtp_diffpure_zys.sh ./results/constrained_32_ddpm_150_150 ./adversarial_images/prompt_constrained_32.bmp 150 150 4
bash minigpt_eval_rtp_diffpure_zys.sh ./results/constrained_64_ddpm_150_150 ./adversarial_images/prompt_constrained_64.bmp 150 150 5
bash minigpt_eval_rtp_diffpure_zys.sh ./results/unconstrained_ddpm_150_150 ./adversarial_images/prompt_unconstrained.bmp 150 150 6
```

#### RAW adversiral images with gaussian noise (sigma=30)
```bash
bash minigpt_eval_rtp_diffpure_zys.sh ./results/clean_G30_ddpm_150_150 ./adversarial_images_add_noise_G30/clean.jpeg 150 150 3
bash minigpt_eval_rtp_diffpure_zys.sh ./results/constrained_16_G30_ddpm_150_150 ./adversarial_images_add_noise_G30/prompt_constrained_16.bmp 150 150 4
bash minigpt_eval_rtp_diffpure_zys.sh ./results/constrained_32_G30_ddpm_150_150 ./adversarial_images_add_noise_G30/prompt_constrained_32.bmp 150 150 5
bash minigpt_eval_rtp_diffpure_zys.sh ./results/constrained_64_G30_ddpm_150_150 ./adversarial_images_add_noise_G30/prompt_constrained_64.bmp 150 150 6
bash minigpt_eval_rtp_diffpure_zys.sh ./results/unconstrained_G30_ddpm_150_150 ./adversarial_images_add_noise_G30/prompt_unconstrained.bmp 150 150 7
```

#### RAW adversiral images with gaussian noise (sigma=50)
```bash
bash minigpt_eval_rtp_diffpure_zys.sh ./results/clean_G50_ddpm_150_150 ./adversarial_images_add_noise_G50/clean.jpeg 150 150 3
bash minigpt_eval_rtp_diffpure_zys.sh ./results/constrained_16_G50_ddpm_150_150 ./adversarial_images_add_noise_G50/prompt_constrained_16.bmp 150 150 4
bash minigpt_eval_rtp_diffpure_zys.sh ./results/constrained_32_G50_ddpm_150_150 ./adversarial_images_add_noise_G50/prompt_constrained_32.bmp 150 150 5
bash minigpt_eval_rtp_diffpure_zys.sh ./results/constrained_64_G50_ddpm_150_150 ./adversarial_images_add_noise_G50/prompt_constrained_64.bmp 150 150 6
bash minigpt_eval_rtp_diffpure_zys.sh ./results/unconstrained_G50_ddpm_150_150 ./adversarial_images_add_noise_G50/prompt_unconstrained.bmp 150 150 7
```

#### RAW adversiral images with gaussian noise (sigma=75)
```bash
bash minigpt_eval_rtp_diffpure_zys.sh ./results/clean_G75_ddpm_150_150 ./adversarial_images_add_noise_G75/clean.jpeg 150 150 3
bash minigpt_eval_rtp_diffpure_zys.sh ./results/constrained_16_G75_ddpm_150_150 ./adversarial_images_add_noise_G75/prompt_constrained_16.bmp 150 150 4
bash minigpt_eval_rtp_diffpure_zys.sh ./results/constrained_32_G75_ddpm_150_150 ./adversarial_images_add_noise_G75/prompt_constrained_32.bmp 150 150 5
bash minigpt_eval_rtp_diffpure_zys.sh ./results/constrained_64_G75_ddpm_150_150 ./adversarial_images_add_noise_G75/prompt_constrained_64.bmp 150 150 6
bash minigpt_eval_rtp_diffpure_zys.sh ./results/unconstrained_G75_ddpm_150_150 ./adversarial_images_add_noise_G75/prompt_unconstrained.bmp 150 150 7
```

