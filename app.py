import uuid
import gradio as gr
from managers.DepthEstimation import DepthEstimation
from managers.ImageCaption import ImageCaption
from managers.ImageGeneration import ImageGeneration
from managers.InterpolationManager import InterpolationManager
from managers.ObjectDetection import ObjectDetection
from pathlib import Path

# object_detection = ObjectDetection()
image_captioning = ImageCaption()
depth_estimation = DepthEstimation()
image_generation = ImageGeneration()
interpolation_manager = InterpolationManager()

data_folder = Path("data")
generated_image_folder = Path(data_folder,"generated_images")
generated_gifs = Path(data_folder,"generated_gifs")

def predict(image, prompt,negative_prompt, override_options, method_options='Depth', intensity_value=0.8):
    print(type(image))
    print(f'The size of the image is {image.size}')
    
    caption = image_captioning.generate_caption(image)

    unique_random_id = str(uuid.uuid4())
    generated_gif_path = generated_gifs / f'generated_gif_{unique_random_id}.gif'
    resisez_image = image_generation._resize_image(image)
    
    if method_options == 'Image':
        estimated_depth = resisez_image
        generated_image = image_generation.generate_image_from_image(image, caption, extended_prompt=prompt,negative_prompt=negative_prompt, override_options=override_options, intensity=intensity_value)
    else:
        generated_image = image_generation.generate_image_from_depth(image, caption, extended_prompt=prompt,negative_prompt=negative_prompt, override_options=override_options)
        estimated_depth = depth_estimation.estimate_depth(image)
    
    # estimated_depth = image
    # generated_image = image_generation.generate_image(caption)
    # generated_image_from_img = image_generation.generate_image_from_image(image, caption)

   
    interpolation_manager.merge_images_using_fade(resisez_image, generated_image, output_gif = generated_gif_path)

    return generated_gif_path, caption, generated_image, estimated_depth

# create gradio app to load image and predict
def create_app():
    # create interface that inputs image (jpeg and png) and outputs label (string)
    input_prompt = gr.inputs.Textbox(lines=2, label='Prompt')
    negative_prompt = gr.inputs.Textbox(lines=2, label='Negative Prompt')
    override_options = gr.inputs.Radio(['Append','Replace','Override'], label='Prompt options', default='Append')
    method_options = gr.inputs.Radio(['Image','Depth'], label='Method', default='Depth')
    intensity_value = gr.inputs.Slider(minimum=0, maximum=1, default=1, label='Intensity')
    input_image = gr.inputs.Image(label='Original Image',type='pil', source='upload')
    output_image = gr.outputs.Image(label='Generated Image',type='pil')
    output_caption = gr.outputs.Textbox(label='Original Image Caption',type='text')
    output_depth_estimation = gr.outputs.Image(label='Predicted depth',type='pil')
    output_generated_img = gr.outputs.Image(label='Generated image',type='pil')
    output_generated_img_from_img = gr.outputs.Image(label='Generated image from image',type='pil')
    output_generated_img_from_depth = gr.outputs.Image(label='Generated image from depth',type='pil')
    app = gr.Interface(
        predict,
        [input_image, input_prompt,negative_prompt, override_options, method_options, intensity_value],
        outputs=[
                output_image,
                output_caption, 
                output_generated_img_from_depth,
                output_depth_estimation, 
                ],
    )

    return app


demo = create_app()
demo.launch(
        debug=False,
        server_name="0.0.0.0",
        share=False,
    )






