import gradio as gr
from managers.DepthEstimation import DepthEstimation
from managers.ImageCaption import ImageCaption
from managers.ImageGeneration import ImageGeneration
from managers.ObjectDetection import ObjectDetection

# object_detection = ObjectDetection()
image_captioning = ImageCaption()
depth_estimation = DepthEstimation()
image_generation = ImageGeneration()

def predict(image, prompt,negative_prompt, override_options):
    print(type(image))
    print(f'The size of the image is {image.size}')

    # detected_objects = object_detection.detect(image)
    detected_objects = image
    caption = image_captioning.generate_caption(image)
    estimated_depth = depth_estimation.estimate_depth(image)
    # estimated_depth = image
    # generated_image = image_generation.generate_image(caption)
    generated_image = image
    # generated_image_from_img = image_generation.generate_image_from_image(image, caption)
    generated_image_from_img = image
    generated_image_from_depth = image_generation.generate_image_from_depth(image, caption, extended_prompt=prompt,negative_prompt=negative_prompt, override_options=override_options)

    return generated_image_from_depth, caption, estimated_depth, generated_image, generated_image_from_img, detected_objects

# create gradio app to load image and predict
def create_app():
    # create interface that inputs image (jpeg and png) and outputs label (string)
    input_prompt = gr.inputs.Textbox(lines=2, label='Prompt')
    negative_prompt = gr.inputs.Textbox(lines=2, label='Negative Prompt')
    override_options = gr.inputs.Radio(['Append','Replace','Override'], label='Prompt options', default='Append')
    input_image = gr.inputs.Image(label='Original Image',type='pil', source='upload')
    output_image = gr.outputs.Image(label='Generated Image',type='pil')
    output_caption = gr.outputs.Textbox(label='Original Image Caption',type='text')
    output_depth_estimation = gr.outputs.Image(label='Predicted depth',type='pil')
    output_generated_img = gr.outputs.Image(label='Generated image',type='pil')
    output_generated_img_from_img = gr.outputs.Image(label='Generated image from image',type='pil')
    output_generated_img_from_depth = gr.outputs.Image(label='Generated image from depth',type='pil')
    app = gr.Interface(
        predict,
        [input_image, input_prompt,negative_prompt, override_options],
        outputs=[output_image,
                output_caption, 
                output_depth_estimation, 
                output_generated_img,
                output_generated_img_from_img,
                output_generated_img_from_depth
                ],
    )

    return app


demo = create_app()
demo.launch(
        debug=False,
        server_name="0.0.0.0",
        share=False,
    )






