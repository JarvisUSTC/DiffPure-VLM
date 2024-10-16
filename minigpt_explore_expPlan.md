# Experiment plan

## Evaluate the Distance of 'Adv. image with Diffusion process' and 'Clean image with Gaussian Noise'

In the test script, the template is 

`bash minigpt_explore.sh {output_dir} {adv_image_path} {constraint} {ddpm_forward_step} {ddpm_reverse_step} {gpu_device_id}`

Please switch to minigpt4 conda environment to run these scripts.

## MiniGPT-4

### Constraint v.s. Steps

#### Adversiral images with constraint 16
```bash
bash minigpt_explore.sh ./explore/Noisy_Dist_Cons_16_Diff_30_30 ./adversarial_images/prompt_constrained_16.bmp 16 30 30 0
bash minigpt_explore.sh ./explore/Noisy_Dist_Cons_16_Diff_50_50 ./adversarial_images/prompt_constrained_16.bmp 16 50 50 0
bash minigpt_explore.sh ./explore/Noisy_Dist_Cons_16_Diff_100_100 ./adversarial_images/prompt_constrained_16.bmp 16 100 100 0
bash minigpt_explore.sh ./explore/Noisy_Dist_Cons_16_Diff_150_150 ./adversarial_images/prompt_constrained_16.bmp 16 150 150 0
bash minigpt_explore.sh ./explore/Noisy_Dist_Cons_16_Diff_200_200 ./adversarial_images/prompt_constrained_16.bmp 16 200 200 0
bash minigpt_explore.sh ./explore/Noisy_Dist_Cons_16_Diff_250_250 ./adversarial_images/prompt_constrained_16.bmp 16 250 250 0
bash minigpt_explore.sh ./explore/Noisy_Dist_Cons_16_Diff_300_300 ./adversarial_images/prompt_constrained_16.bmp 16 300 300 0
bash minigpt_explore.sh ./explore/Noisy_Dist_Cons_16_Diff_350_350 ./adversarial_images/prompt_constrained_16.bmp 16 350 350 0
bash minigpt_explore.sh ./explore/Noisy_Dist_Cons_16_Diff_400_400 ./adversarial_images/prompt_constrained_16.bmp 16 400 400 0
bash minigpt_explore.sh ./explore/Noisy_Dist_Cons_16_Diff_450_450 ./adversarial_images/prompt_constrained_16.bmp 16 450 450 0
bash minigpt_explore.sh ./explore/Noisy_Dist_Cons_16_Diff_500_500 ./adversarial_images/prompt_constrained_16.bmp 16 500 500 0
bash minigpt_explore.sh ./explore/Noisy_Dist_Cons_16_Diff_550_550 ./adversarial_images/prompt_constrained_16.bmp 16 550 550 0
bash minigpt_explore.sh ./explore/Noisy_Dist_Cons_16_Diff_600_600 ./adversarial_images/prompt_constrained_16.bmp 16 600 600 0
bash minigpt_explore.sh ./explore/Noisy_Dist_Cons_16_Diff_650_650 ./adversarial_images/prompt_constrained_16.bmp 16 650 650 0
bash minigpt_explore.sh ./explore/Noisy_Dist_Cons_16_Diff_700_700 ./adversarial_images/prompt_constrained_16.bmp 16 700 700 0
bash minigpt_explore.sh ./explore/Noisy_Dist_Cons_16_Diff_750_750 ./adversarial_images/prompt_constrained_16.bmp 16 750 750 0
```




#### Adversiral images with constraint 32
```bash
bash minigpt_explore.sh ./explore/Noisy_Dist_Cons_32_Diff_30_30 ./adversarial_images/prompt_constrained_32.bmp 32 30 30 0
bash minigpt_explore.sh ./explore/Noisy_Dist_Cons_32_Diff_50_50 ./adversarial_images/prompt_constrained_32.bmp 32 50 50 0
bash minigpt_explore.sh ./explore/Noisy_Dist_Cons_32_Diff_100_100 ./adversarial_images/prompt_constrained_32.bmp 32 100 100 0
bash minigpt_explore.sh ./explore/Noisy_Dist_Cons_32_Diff_150_150 ./adversarial_images/prompt_constrained_32.bmp 32 150 150 0
bash minigpt_explore.sh ./explore/Noisy_Dist_Cons_32_Diff_200_200 ./adversarial_images/prompt_constrained_32.bmp 32 200 200 0
bash minigpt_explore.sh ./explore/Noisy_Dist_Cons_32_Diff_250_250 ./adversarial_images/prompt_constrained_32.bmp 32 250 250 0
bash minigpt_explore.sh ./explore/Noisy_Dist_Cons_32_Diff_300_300 ./adversarial_images/prompt_constrained_32.bmp 32 300 300 0
bash minigpt_explore.sh ./explore/Noisy_Dist_Cons_32_Diff_350_350 ./adversarial_images/prompt_constrained_32.bmp 32 350 350 0
bash minigpt_explore.sh ./explore/Noisy_Dist_Cons_32_Diff_400_400 ./adversarial_images/prompt_constrained_32.bmp 32 400 400 0
bash minigpt_explore.sh ./explore/Noisy_Dist_Cons_32_Diff_450_450 ./adversarial_images/prompt_constrained_32.bmp 32 450 450 0
bash minigpt_explore.sh ./explore/Noisy_Dist_Cons_32_Diff_500_500 ./adversarial_images/prompt_constrained_32.bmp 32 500 500 0
bash minigpt_explore.sh ./explore/Noisy_Dist_Cons_32_Diff_550_550 ./adversarial_images/prompt_constrained_32.bmp 32 550 550 0
bash minigpt_explore.sh ./explore/Noisy_Dist_Cons_32_Diff_600_600 ./adversarial_images/prompt_constrained_32.bmp 32 600 600 0
bash minigpt_explore.sh ./explore/Noisy_Dist_Cons_32_Diff_650_650 ./adversarial_images/prompt_constrained_32.bmp 32 650 650 0
bash minigpt_explore.sh ./explore/Noisy_Dist_Cons_32_Diff_700_700 ./adversarial_images/prompt_constrained_32.bmp 32 700 700 0
bash minigpt_explore.sh ./explore/Noisy_Dist_Cons_32_Diff_750_750 ./adversarial_images/prompt_constrained_32.bmp 32 750 750 0
```



#### Adversiral images with constraint 64
```bash
bash minigpt_explore.sh ./explore/Noisy_Dist_Cons_64_Diff_30_30 ./adversarial_images/prompt_constrained_64.bmp 64 30 30 0
bash minigpt_explore.sh ./explore/Noisy_Dist_Cons_64_Diff_50_50 ./adversarial_images/prompt_constrained_64.bmp 64 50 50 0
bash minigpt_explore.sh ./explore/Noisy_Dist_Cons_64_Diff_100_100 ./adversarial_images/prompt_constrained_64.bmp 64 100 100 0
bash minigpt_explore.sh ./explore/Noisy_Dist_Cons_64_Diff_150_150 ./adversarial_images/prompt_constrained_64.bmp 64 150 150 0
bash minigpt_explore.sh ./explore/Noisy_Dist_Cons_64_Diff_200_200 ./adversarial_images/prompt_constrained_64.bmp 64 200 200 0
bash minigpt_explore.sh ./explore/Noisy_Dist_Cons_64_Diff_250_250 ./adversarial_images/prompt_constrained_64.bmp 64 250 250 0
bash minigpt_explore.sh ./explore/Noisy_Dist_Cons_64_Diff_300_300 ./adversarial_images/prompt_constrained_64.bmp 64 300 300 0
bash minigpt_explore.sh ./explore/Noisy_Dist_Cons_64_Diff_350_350 ./adversarial_images/prompt_constrained_64.bmp 64 350 350 0
bash minigpt_explore.sh ./explore/Noisy_Dist_Cons_64_Diff_400_400 ./adversarial_images/prompt_constrained_64.bmp 64 400 400 0
bash minigpt_explore.sh ./explore/Noisy_Dist_Cons_64_Diff_450_450 ./adversarial_images/prompt_constrained_64.bmp 64 450 450 0
bash minigpt_explore.sh ./explore/Noisy_Dist_Cons_64_Diff_500_500 ./adversarial_images/prompt_constrained_64.bmp 64 500 500 0
bash minigpt_explore.sh ./explore/Noisy_Dist_Cons_64_Diff_550_550 ./adversarial_images/prompt_constrained_64.bmp 64 550 550 0
bash minigpt_explore.sh ./explore/Noisy_Dist_Cons_64_Diff_600_600 ./adversarial_images/prompt_constrained_64.bmp 64 600 600 0
bash minigpt_explore.sh ./explore/Noisy_Dist_Cons_64_Diff_650_650 ./adversarial_images/prompt_constrained_64.bmp 64 650 650 0
bash minigpt_explore.sh ./explore/Noisy_Dist_Cons_64_Diff_700_700 ./adversarial_images/prompt_constrained_64.bmp 64 700 700 0
bash minigpt_explore.sh ./explore/Noisy_Dist_Cons_64_Diff_750_750 ./adversarial_images/prompt_constrained_64.bmp 64 750 750 0
```
