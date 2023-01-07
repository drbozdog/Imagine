import os
blip_project_path = os.path.join('..','BLIP')
os.sys.path.append(blip_project_path)
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from models.blip import blip_decoder

class ImageCaption():
    def __init__(self):
        self.device = 'cpu'

        model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'

        self.models = {
            'image_captioning': blip_decoder(pretrained=model_url,
                                             image_size=384, vit='base')
        }

    def generate_caption(self,image):
        task = 'image_captioning'

        im = load_image(image, image_size=480 if task == 'visual_question_answering' else 384, device=self.device)
        model = self.models[task]
        model.eval()
        model = model.to(self.device)

        if task == 'image_captioning':
            with torch.no_grad():
                caption = model.generate(im, sample=False, num_beams=3, max_length=20, min_length=5)
                return caption[0]
        

def load_image(image, image_size, device):
    raw_image = image

    w, h = raw_image.size

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    image = transform(raw_image).unsqueeze(0).to(device)
    return image
