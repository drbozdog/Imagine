import torch
import transformers
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
from PIL import Image
import numpy as np

class DepthEstimation():
    def __init__(self):
        model_path = 'Intel/dpt-large'
        self.model = DPTForDepthEstimation.from_pretrained(model_path)
        self.model = self.model.to('cpu')
        self.feature_extractor = DPTFeatureExtractor.from_pretrained(model_path)

    def estimate_depth(self, image):
        inputs = self.feature_extractor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        # interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        )

        # visualize the prediction
        output = prediction.squeeze().cpu().numpy()
        formatted = (output * 255 / np.max(output)).astype("uint8")
        depth = Image.fromarray(formatted)

        print(f'Estimated depth: {depth.size}')

        return depth
