
import uuid
from PIL import Image
from diffusers import StableDiffusionDepth2ImgPipeline, StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
import torch
from pathlib import Path
import os

class ImageGeneration():
    def __init__(self) -> None:
        self.model_id = "stabilityai/stable-diffusion-2"
        self.steps = 40
        self.negative_prompts = ''
        self.positive_prompts = ''
        self.height = 768
        self.width = 768

        self.generated_image_folder = Path("data","generated_images")
        self.current_method = None
        self.pipeline = None


    def get_pipeline(self, method='Depth'):
        # using process call nvidia-smi and print the output
        print(os.system('nvidia-smi'))
        if self.current_method != method:
            if torch.cuda.is_available():
                if self.pipeline:
                    del self.pipeline
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                print("GPU cache cleared")
                print(os.system('nvidia-smi'))
            if method == 'Depth':
                pipeline = StableDiffusionDepth2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-depth")
                pipeline.to('cuda')
                pipeline.enable_attention_slicing()
            elif method == 'Image':
                pipeline = StableDiffusionImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")
                pipeline.to('cuda')
                pipeline.enable_attention_slicing()
            self.current_method = method
            self.pipeline = pipeline
            return pipeline
        else:
            return self.pipeline
            


    def get_prompts_for_style(style='photography'):
        if style == 'photography':
            pos_prompt = "A photography, epic, exciting, wow, cinematic, moody, exciting, stop motion, highly detailed, octane render, soft lighting, professional, 35mm, Zeiss, Hasselblad, Fujifilm, Arriflex, IMAX, 4k, 8k"
            neg_prompt = "bright, oversaturated, ugly, 3d, render, cartoon, grain, low-res, kitsch, blender, cropped, lowres, poorly drawn face, out of frame, poorly drawn hands, blurry, bad art, blurred, text, watermark, disfigured, deformed, mangled"
        elif style == 'painting':
            pos_prompt = ''
            neg_prompt = ''
        elif style == 'drawing':
            pos_prompt = ''
            neg_prompt = ''
        elif style == 'cartoon':
            pos_prompt = ''
            neg_prompt = ''
        elif style == 'anime':
            pos_prompt = ''
            neg_prompt = ''
        elif style == 'sketch':
            pos_prompt = ''
            neg_prompt = ''
        elif style == 'illustration':
            pos_prompt = ''
            neg_prompt = ''
        elif style == 'abstract':
            pos_prompt = ''
            neg_prompt = ''
        elif style == '3d':
            pos_prompt = ''
            neg_prompt = ''
        elif style == 'pixel':
            pos_prompt = ''
            neg_prompt = ''
        elif style == 'vector':
            pos_prompt = ''
            neg_prompt = ''
        elif style == 'manga':
            pos_prompt = ''
            neg_prompt = ''
        elif style == 'comic':
            pos_prompt = ''
            neg_prompt = ''
        elif style == 'realistic':
            pos_prompt = ''
            neg_prompt = ''
        else:
            raise ValueError('Invalid style')
        
        

    def _ensemble_prompt(self, text, extended_prompt,negative_prompt,override_options):
        style_prompt = ", epic, exciting, wow, cinematic, moody, exciting, stop motion, highly detailed, octane render, soft lighting, professional, 35mm, Zeiss, Hasselblad, Fujifilm, Arriflex, IMAX, 4k, 8k"
        neg_prompt = "bright, oversaturated, ugly, 3d, render, cartoon, grain, low-res, kitsch, blender, cropped, lowres, poorly drawn face, out of frame, poorly drawn hands, blurry, bad art, blurred, text, watermark, disfigured, deformed, mangled"
        if override_options == 'Append':
            text = text + " " + extended_prompt + style_prompt
        elif override_options == 'Replace':
            text = extended_prompt + style_prompt
        elif override_options == 'Override':
            text = extended_prompt
            neg_prompt = negative_prompt

        print(f"Prompt: {text}")
        print(f"Negative Prompt: {neg_prompt}")

        return text, neg_prompt


    def generate_image(self, text):

        orig_prompt = "A "+text+", epic, exciting, wow, cinematic, moody, exciting, stop motion, highly detailed, octane render, soft lighting, professional, 35mm, Zeiss, Hasselblad, Fujifilm, Arriflex, IMAX, 4k, 8k"
        orig_negative_prompt = "bright, oversaturated, ugly, 3d, render, cartoon, grain, low-res, kitsch, blender, cropped, lowres, poorly drawn face, out of frame, poorly drawn hands, blurry, bad art, blurred, text, watermark, disfigured, deformed, mangled"
        pipeline = StableDiffusionPipeline.from_pretrained(self.model_id)
        pipeline.to('mps')
        return pipeline(orig_prompt,
            height = self.height, 
            width = self.width,
            num_inference_steps = self.steps,
            negative_prompt = orig_negative_prompt,
        ).images[0]

    def generate_image_from_image(self, image,text, extended_prompt,negative_prompt, override_options, intensity=0.7):
        orig_prompt, orig_negative_prompt = self._ensemble_prompt(text, extended_prompt,negative_prompt, override_options)

        
        resized_image = self._resize_image(image)
        
        generated_image =  self.get_pipeline('Image')(
            prompt=orig_prompt,
            image=resized_image,
            num_inference_steps=self.steps,
            negative_prompt=orig_negative_prompt,
            strength=intensity
        ).images[0]
        print(f'Generated image: {generated_image.size}')
        return generated_image

    def generate_image_from_depth(self, image, text, extended_prompt,negative_prompt,override_options):
    
        orig_prompt, orig_negative_prompt = self._ensemble_prompt(text, extended_prompt,negative_prompt, override_options)

        print(f'Original prompt: {orig_prompt}')
        print(f'Original negative prompt: {orig_negative_prompt}')

        print(f'Original image: {image.size}')
        
        init_image = self._resize_image(image)

        # generate random experiment id
        experiment_id = str(uuid.uuid4())

        # save the initial image to file
        init_image.save(self.generated_image_folder/ f'initial_image_{experiment_id}.png')
        
        image = self.get_pipeline('Depth')(prompt=orig_prompt, image=init_image, negative_prompt=orig_negative_prompt, strength=1, 
            num_inference_steps=self.steps
            ).images[0]
        print(f'Generated image: {image.size}')

        image.save(self.generated_image_folder / f'generated_image_{experiment_id}.png')

        return image

    
    def _resize_image(self, image):
        width, height = image.size
        image = image.copy()
        # Calculate the new width and height of the image such that the largest dimension is 768
        if width > height:
            new_width = self.width
            new_height = int(self.width * height / width)
        else:
            new_width = int(self.height * width / height)
            new_height = self.height

        print(f'Original image: {image.size}')
        print(f'New image: {new_width}x{new_height}')
        # crop from the center using the new width and height
        image = image.resize((new_width, new_height), Image.ANTIALIAS)

        new_height = new_height - (new_height % 64)
        new_width = new_width - (new_width % 64)

        # Get the size of the image
        width, height = image.size

        # Calculate the center point
        center_x, center_y = width // 2, height // 2

        # Calculate the dimensions of the cropped image
        left, top, right, bottom = center_x - new_width // 2, center_y - new_height // 2, center_x + new_width // 2, center_y + new_height // 2

        # Crop the image
        cropped_image = image.crop((left, top, right, bottom))
        print(f'Cropped image: {cropped_image.size}')

        return cropped_image