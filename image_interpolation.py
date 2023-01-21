from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel
import torch
import numpy as np
import lovely_tensors as lt
from tqdm import tqdm

lt.monkey_patch()

model_name = 'stabilityai/stable-diffusion-2-1'
weight_dtype = torch.float32


vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")
vae.requires_grad_(False)

def convert_pil_image_from_file_to_tensor(file):
    from PIL import Image
    image = Image.open(file)
    image = np.array(image) / 255
    # convert the numpy array to torch tensor
    image = torch.tensor(image).to(dtype=weight_dtype)
    # Add a batch dimension to the tensor
    image = image.permute(2, 0, 1).unsqueeze(0)
    return image

# load image from jpg file using PIL
from PIL import Image
start_image = convert_pil_image_from_file_to_tensor('67053_RIDGID_RP350 Press Tool_right jaw_72dpi (1).jpeg')
end_image = convert_pil_image_from_file_to_tensor('RP 350 C 005_72dpi.jpeg')

start_latents = vae.encode(start_image.to(dtype=weight_dtype)).latent_dist.sample()
end_latents = vae.encode(end_image.to(dtype=weight_dtype)).latent_dist.sample()

steps = 30
for i in tqdm(range(steps),total=steps):
    # interpolate between the two images
    interpolated_latents = torch.lerp(start_latents, end_latents, i/steps)

    # decode the interpolated latents
    interpolated_image = vae.decode(interpolated_latents)

    # convert the torch tensor image to a PIL image
    from torchvision.transforms import ToPILImage
    to_pil = ToPILImage()
    pil_image = to_pil(interpolated_image.sample[0].cpu())
    # save the image to a file
    pil_image.save(f'interpolated_image_{i}.jpg')


# create gif from the images
import imageio
images = []
for i in range(steps):
    images.append(imageio.imread(f'interpolated_image_{i}.jpg'))
    imageio.mimsave('interpolated_image.gif', images)

