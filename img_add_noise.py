import os
import numpy as np
from PIL import Image

adv_img_folder = './adversarial_images'
adv_img_noise_folder = './adversarial_images_add_noise_G75'

adv_img_names = os.listdir(adv_img_folder)
print('adv_imgs: ', adv_img_names)

for adv_img_name in adv_img_names:
    if adv_img_name.split('.')[1] in ['bmp', 'jpeg']:
        adv_img = Image.open(os.path.join(adv_img_folder, adv_img_name))
        adv_img_array = np.array(adv_img)

        noise = np.random.normal(0, 75, adv_img_array.shape)

        noisy_adv_img_arr = np.clip(adv_img_array + noise, 0, 255).astype('uint8')

        noisy_adv_img = Image.fromarray(noisy_adv_img_arr)
        noisy_adv_img.save(os.path.join(adv_img_noise_folder, adv_img_name))
