from transformers import AutoImageProcessor, AutoModelForObjectDetection
import torch

class ObjectDetection():

    def __init__(self):
        self.image_processor = AutoImageProcessor.from_pretrained("hustvl/yolos-tiny")
        self.model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-tiny")

    def detect(self,image):
        image = image.copy()
        inputs = self.image_processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        results = self.image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]

        def draw_boxes(image, boxes_tensors, labels):
            from PIL import ImageDraw
            draw = ImageDraw.Draw(image)
            for box_tensor, label in zip(boxes_tensors, labels):
                box = box_tensor.tolist()
                label = self.model.config.id2label[label.item()]
                draw.rectangle(box, outline="red", width=3)
                draw.text(box[:2], label, fill="red")
            return image

        image_with_boxes = draw_boxes(image, results["boxes"], results["labels"])

        return image_with_boxes

