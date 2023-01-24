import imageio
from PIL import Image

from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel
import torch
import numpy as np
import lovely_tensors as lt
from tqdm import tqdm

lt.monkey_patch()

model_name = 'stabilityai/stable-diffusion-2-1'
weight_dtype = torch.float32

class InterpolationManager():
    def __init__(self):
        self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")
        self.vae.requires_grad_(False)
        self.vae.to('cpu')

    def merge_images_using_fade(self, start_image, end_image, output_gif, steps=30):
            # start_image = Image.open(start_image)
            # end_image = Image.open(end_image)

            # create list of alphas to use for blending
            alphas = np.linspace(0, 1, steps).tolist()
            alphas+=[1]*4
            alphas += reversed(alphas)
            print(alphas)

            images = []
            for alpha in alphas:
                # create the blended image
                blended = Image.blend(start_image, end_image, alpha)
                # add the blended image to the list
                images.append(blended)
            
            imageio.mimsave(output_gif, images)

    def interpolate_using_vae(self,start_image, end_image, output_gif, steps = 30):
        start_image = self.convert_pil_image_from_file_to_tensor(start_image)
        end_image = self.convert_pil_image_from_file_to_tensor(end_image)

        start_latents = self.vae.encode(start_image.to(dtype=weight_dtype).to('cuda')).latent_dist.sample()
        end_latents = self.vae.encode(end_image.to(dtype=weight_dtype).to('cuda')).latent_dist.sample()

        images = []

        for i in tqdm(range(steps),total=steps):
            # interpolate between the two images
            interpolated_latents = torch.lerp(start_latents, end_latents, i/steps)

            # decode the interpolated latents
            interpolated_image = self.vae.decode(interpolated_latents)

            # convert the torch tensor image to a PIL image
            from torchvision.transforms import ToPILImage
            to_pil = ToPILImage()
            pil_image = to_pil(interpolated_image.sample[0].cpu())
            # save the image to a file
            images.append(pil_image)

        imageio.mimsave(output_gif, images)

    
    # Define the sigmoid function
    def sigmoid(self,x, steepness=10, midpoint=0.5):
        return 1 / (1 + np.exp(-steepness * (x - midpoint)))


    def convert_pil_image_from_file_to_tensor(self, file):
        image = Image.open(file)
        image = np.array(image) / 255
        # convert the numpy array to torch tensor
        image = torch.tensor(image).to(dtype=weight_dtype)
        # Add a batch dimension to the tensor
        image = image.permute(2, 0, 1).unsqueeze(0)
        return image


if __name__=="__main__":
    # load image from jpg file using PIL
    start_image = '/home/tomitza/Projects/Imagine/data/generated_images/initial_image_a4eaecfd-9bb6-407f-a268-62d7bbf57aaf.png'
    end_image = '/home/tomitza/Projects/Imagine/data/generated_images/generated_image_a2c7bfdb-f99e-4fbe-a60d-fc6bfe63c1a4.png'

    start_image = Image.open(start_image)
    end_image = Image.open(end_image)

    interpolation_manager = InterpolationManager()
    images = interpolation_manager.merge_images_using_fade(start_image, end_image, 'interpolated.gif')

    # images = interpolation_manager.interpolate_using_vae(start_image, end_image)
    # interpolation_manager.create_gif(images, 'interpolated.gif')