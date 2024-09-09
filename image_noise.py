import os
import numpy as np
from PIL import Image

adv_noise_folder = './images_noise'

noiseType = 'Gaussian'
sigma = 0

noise = np.random.normal(0, sigma, (224, 224, 3))
noise_sign = np.abs(noise)

noisy_adv_img_arr = np.clip(noise, 0, 255).astype('uint8')
noisy_adv_img = Image.fromarray(noisy_adv_img_arr)
adv_noise_name = noiseType + '_sigma_' + str(sigma) + '.bmp'
noisy_adv_img.save(os.path.join(adv_noise_folder, adv_noise_name))

noisy_sign_adv_img_arr = np.clip(noise_sign, 0, 255).astype('uint8')
noisy_sign_adv_img = Image.fromarray(noisy_sign_adv_img_arr)
adv_noise_sign_name = noiseType + '_sigma_' + str(sigma) + '_signed.bmp'
noisy_sign_adv_img.save(os.path.join(adv_noise_folder, adv_noise_sign_name))
