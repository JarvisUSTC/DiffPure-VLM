import os
import torch
import argparse
from PIL import Image
from load_diffusion_model import load_diffusion_models
from purification import PurificationForward
# from minigpt4.common.config import Config
# from minigpt4.common.registry import registry
import torch.nn as nn
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from lavis.models.eva_vit import create_eva_vit_g
from lavis.models.clip_vit import create_clip_vit_L
from lavis.models.blip2_models.blip2 import disabled_train


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default="eval_configs/minigpt4_eval.yaml", help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    
    parser.add_argument("--clean_image", type=str, default='./image.bmp',
                        help="Clean Image file")
    parser.add_argument("--image_file", type=str, default='./image.bmp',
                        help="Image file")
    parser.add_argument("--output_folder", type=str, default='./test',
                        help="Output file.")
    parser.add_argument("--constraint", type=int, default=0, help="The constraint to generate adversarial image.")

    # Purification hyperparameters in defense
    parser.add_argument("--def_max_timesteps", type=str, required=True,
                        help='The number of forward steps for each purification step in defense')
    parser.add_argument('--def_num_denoising_steps', type=str, required=True,
                        help='The number of denoising steps for each purification step in defense')
    parser.add_argument('--def_sampling_method', type=str, default='ddpm', choices=['ddpm', 'ddim'],
                        help='Sampling method for the purification in defense')
    
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args

def get_diffusion_params(max_timesteps, num_denoising_steps):
    max_timestep_list = [int(i) for i in max_timesteps.split(',')]
    num_denoising_steps_list = [int(i) for i in num_denoising_steps.split(',')]
    assert len(max_timestep_list) == len(num_denoising_steps_list)

    diffusion_steps = []
    for i in range(len(max_timestep_list)):
        diffusion_steps.append([i - 1 for i in range(max_timestep_list[i] // num_denoising_steps_list[i],
                               max_timestep_list[i] + 1, max_timestep_list[i] // num_denoising_steps_list[i])])
        max_timestep_list[i] = max_timestep_list[i] - 1

    return max_timestep_list, diffusion_steps

def normalize(images):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()
    images = images - mean[None, :, None, None]
    images = images / std[None, :, None, None]
    return images

def denormalize(images):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()
    images = images * std[None, :, None, None]
    images = images + mean[None, :, None, None]
    return images

args = parse_args()
# cfg = Config(args)

# vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
# vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

def_max_timesteps, def_diffusion_steps = get_diffusion_params(
        args.def_max_timesteps, args.def_num_denoising_steps)

device = 'cuda:{}'.format(args.gpu_id)

print('def_max_timesteps: ', def_max_timesteps)
print('def_diffusion_steps: ', def_diffusion_steps)
print('def_sampling_method: ', args.def_sampling_method)

diffusion = load_diffusion_models(args, 'ckpts/diffpure_models/diffusion/Guide_Diffusion/256x256_diffusion_uncond.pt', device)
print('Initialization Finished')

# Data processing in pixel space
vis_processor_pixel = T.Compose(
    [
        T.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
    ]
)

# Data processing in embedding space (Overall data processing for embedding space should include the former pixel part)
mean = (0.48145466, 0.4578275, 0.40821073)
std = (0.26862954, 0.26130258, 0.27577711)
vis_processor_embed = T.Normalize(mean, std)


diffusion_process = PurificationForward(diffusion, def_max_timesteps, def_diffusion_steps, args.def_sampling_method, True, device, debug=False, explore=True) # preprocess the image with the diffusion model

ToImage = T.ToPILImage()

adv_img = Image.open(args.image_file).convert('RGB')
adv_img_tensor = vis_processor_pixel(adv_img).unsqueeze(0).to(device)
noised_img, diffused_img = diffusion_process(adv_img_tensor)

# test
# noised_img = adv_img_tensor
# diffused_img = adv_img_tensor

clean_img = Image.open(args.clean_image).convert('RGB')
clean_img_align = vis_processor_pixel(clean_img).unsqueeze(0).to(device) # [0, 1]

forward_steps = def_max_timesteps[0]
t = (torch.ones(adv_img_tensor.size(0)) * forward_steps).to(device)
noise_1 = torch.randn_like(clean_img_align) * (args.constraint / (3 * 255))
noise_2 = torch.randn_like(clean_img_align) * (1.0 - diffusion_process.compute_alpha(t.long())).sqrt()


noisy_image_1 = (clean_img_align + noise_1).clamp(0,1)
noisy_image_2 = (clean_img_align + noise_1 + noise_2).clamp(0,1)

l2_loss = torch.nn.MSELoss()

# For distance calculation in pixel space, all images are normalized to [0, 1], tensor, cuda.
# Calculate the MSE distance of clean image and adv. img
MSE_Dist_0 = l2_loss(clean_img_align, adv_img_tensor)
print('MSE_Dist_0 (clean image and adv. image): {}'.format(MSE_Dist_0))

# Calculate the MSE distance of 'noisy_image_1' (clean image add noise_1) and adv. img
MSE_Dist_1 = l2_loss(noisy_image_1, adv_img_tensor)
print('MSE_Dist_1 (MSE distance of noisy_image_1 (clean image add noise_1) and adv. image): {}'.format(MSE_Dist_1))

# Calculate the MSE distance of 'noisy_image_2' (clean image add noise_1 and noise_2) and adv. img
MSE_Dist_2 = l2_loss(noisy_image_2, adv_img_tensor)
print('MSE_Dist_2 (MSE distance of noisy_image_2 (clean image add noise_1 and noise_2) and adv. image): {}'.format(MSE_Dist_2))

# Calculate the MSE distance of 'noisy_image_1' (clean image add noise_1) and 'diffused_img' (adv. image f and r)
MSE_Dist_3 = l2_loss(noisy_image_1, diffused_img)
print('MSE_Dist_3 with F {} R {} (MSE distance of noisy_image_1 (clean image add noise_1) and diffused_img (adv. image f and r)): {}'.format(def_max_timesteps[0] + 1, def_diffusion_steps[0][-1] + 1, MSE_Dist_3))

# Calculate the MSE distance of 'noisy_image_2' (clean image add noise_1 and noise_2) and 'noised_img' (adv. image f)
MSE_Dist_4 = l2_loss(noisy_image_2, noised_img)
print('MSE_Dist_4 with F {} R {} (MSE distance of noisy_image_2 (clean image add noise_1 and noise_2) and noised_img (adv. image f)): {}'.format(def_max_timesteps[0] + 1, def_diffusion_steps[0][-1] + 1, MSE_Dist_4))


# For distance calculation in embedding space, all images should be normalized based on visual encoder.
# init vision encoder first
def get_embed_from_vision_encoder(input_image, model_name="eva_clip_g", img_size=224, drop_path_rate=0, use_grad_checkpoint=False, precision="fp16"):
    assert model_name in [
        "eva_clip_g",
        "eva2_clip_L",
        "clip_L",
    ], "vit model must be eva_clip_g, eva2_clip_L or clip_L"
    if model_name == "eva_clip_g":
        visual_encoder = create_eva_vit_g(
            img_size, drop_path_rate, use_grad_checkpoint, precision
        )
#         elif model_name == "eva2_clip_L":
#             visual_encoder = create_eva2_vit_L(
#                 img_size, drop_path_rate, use_grad_checkpoint, precision
#             )
    elif model_name == "clip_L":
        visual_encoder = create_clip_vit_L(img_size, use_grad_checkpoint, precision)
    ln_vision = LayerNorm(visual_encoder.num_features)

    for name, param in visual_encoder.named_parameters():
        param.requires_grad = False
    visual_encoder = visual_encoder.eval()
    visual_encoder.train = disabled_train
    image_embeds = ln_vision(visual_encoder(input_image))
    return image_embeds


adv_image_embeds = get_embed_from_vision_encoder(vis_processor_embed(adv_img_tensor))
noised_img_embeds = get_embed_from_vision_encoder(vis_processor_embed(noised_img))
diffused_img_embeds = get_embed_from_vision_encoder(vis_processor_embed(diffused_img))
clean_image_embeds = get_embed_from_vision_encoder(vis_processor_embed(clean_img_align))
noisy_image_1_embeds = get_embed_from_vision_encoder(vis_processor_embed(noisy_image_1))
noisy_image_2_embeds = get_embed_from_vision_encoder(vis_processor_embed(noisy_image_2))

# Calculate the MSE distance of clean image and adv. img in embedding space
MSE_Dist_5 = l2_loss(clean_image_embeds, adv_image_embeds)
print('MSE_Dist_5 (clean image and adv. image) in embedding space: {}'.format(MSE_Dist_5))

# Calculate the MSE distance of 'noisy_image_1' (clean image add noise_1) and adv. img in embedding space
MSE_Dist_6 = l2_loss(noisy_image_1_embeds, adv_image_embeds)
print('MSE_Dist_6 (MSE distance of noisy_image_1 (clean image add noise_1) and adv. image) in embedding space: {}'.format(MSE_Dist_6))

# Calculate the MSE distance of 'noisy_image_2' (clean image add noise_1 and noise_2) and adv. img in embedding space
MSE_Dist_7 = l2_loss(noisy_image_2_embeds, adv_image_embeds)
print('MSE_Dist_7 (MSE distance of noisy_image_2 (clean image add noise_1 and noise_2) and adv. image) in embedding space: {}'.format(MSE_Dist_7))

# Calculate the MSE distance of 'noisy_image_1' (clean image add noise_1) and 'diffused_img' (adv. image f and r) in embedding space
MSE_Dist_8 = l2_loss(noisy_image_1_embeds, diffused_img_embeds)
print('MSE_Dist_8 with F {} R {} (MSE distance of noisy_image_1 (clean image add noise_1) and diffused_img (adv. image f and r)) in embedding space: {}'.format(def_max_timesteps[0] + 1, def_diffusion_steps[0][-1] + 1, MSE_Dist_8))

# Calculate the MSE distance of 'noisy_image_2' (clean image add noise_1 and noise_2) and 'noised_img' (adv. image f) in embedding space
MSE_Dist_9 = l2_loss(noisy_image_2_embeds, noised_img_embeds)
print('MSE_Dist_9 with F {} R {} (MSE distance of noisy_image_2 (clean image add noise_1 and noise_2) and noised_img (adv. image f)) in embedding space: {}'.format(def_max_timesteps[0] + 1, def_diffusion_steps[0][-1] + 1, MSE_Dist_9))




# Output metrics
with open(os.path.join(args.output_folder, 'result.txt'), 'a') as f:
    f.write('MSE_Dist_0 (clean image and adv. image): {}'.format(MSE_Dist_0) + '\n')
    f.write('MSE_Dist_1 (MSE distance of noisy_image_1 (clean image add noise_1) and adv. image): {}'.format(MSE_Dist_1) + '\n')
    f.write('MSE_Dist_2 (MSE distance of noisy_image_2 (clean image add noise_1 and noise_2) and adv. image): {}'.format(MSE_Dist_2) + '\n')
    f.write('MSE_Dist_3 with F {} R {} (MSE distance of noisy_image_1 (clean image add noise_1) and diffused_img (adv. image f and r)): {}'.format(def_max_timesteps[0] + 1, def_diffusion_steps[0][-1] + 1, MSE_Dist_3) + '\n')
    f.write('MSE_Dist_4 with F {} R {} (MSE distance of noisy_image_2 (clean image add noise_1 and noise_2) and noised_img (adv. image f)): {}'.format(def_max_timesteps[0] + 1, def_diffusion_steps[0][-1] + 1, MSE_Dist_4) + '\n')
    f.write('MSE_Dist_5 (clean image and adv. image) in embedding space: {}'.format(MSE_Dist_5) + '\n')
    f.write('MSE_Dist_6 (MSE distance of noisy_image_1 (clean image add noise_1) and adv. image) in embedding space: {}'.format(MSE_Dist_6) + '\n')
    f.write('MSE_Dist_7 (MSE distance of noisy_image_2 (clean image add noise_1 and noise_2) and adv. image) in embedding space: {}'.format(MSE_Dist_7) + '\n')
    f.write('MSE_Dist_8 with F {} R {} (MSE distance of noisy_image_1 (clean image add noise_1) and diffused_img (adv. image f and r)) in embedding space: {}'.format(def_max_timesteps[0] + 1, def_diffusion_steps[0][-1] + 1, MSE_Dist_8) + '\n')
    f.write('MSE_Dist_9 with F {} R {} (MSE distance of noisy_image_2 (clean image add noise_1 and noise_2) and noised_img (adv. image f)) in embedding space: {}'.format(def_max_timesteps[0] + 1, def_diffusion_steps[0][-1] + 1, MSE_Dist_9) + '\n')
    

# Output related images
noised_img_out = ToImage(noised_img.squeeze(0).cpu())
noised_img_out.save(os.path.join(args.output_folder, 'Adv_img_Diff_F.jpg'))
diffused_img_out = ToImage(diffused_img.squeeze(0).cpu())
diffused_img_out.save(os.path.join(args.output_folder, 'Adv_img_Diff_F_Diff_R.jpg'))
noisy_image_1_out = ToImage(noisy_image_1.squeeze(0).cpu())
noisy_image_1_out.save(os.path.join(args.output_folder, 'Clean_img_add_noise_2.jpg'))
noisy_image_2_out = ToImage(noisy_image_2.squeeze(0).cpu())
noisy_image_2_out.save(os.path.join(args.output_folder, 'Clean_img_add_noise_2_noise_1.jpg'))

