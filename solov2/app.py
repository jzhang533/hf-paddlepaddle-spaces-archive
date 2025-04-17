import tempfile
import os

from PIL import Image
import gradio as gr
import paddlehub as hub


module = hub.Module(name="solov2")
    
def inference(img, threshold):
  with tempfile.TemporaryDirectory() as tempdir_name:
    module.predict(image=img, threshold=threshold, visualization=True, save_dir=tempdir_name)
    result_names = os.listdir(tempdir_name)
    output_image = Image.open(os.path.join(tempdir_name, result_names[0]))
  return [output_image]

  
title="SOLOv2"
description="SOLOv2 is a fast instance segmentation model based on paper \"SOLOv2: Dynamic, Faster and Stronger\". The model improves the detection performance and efficiency of masks compared to SOLOv1, and performs well in instance segmentation tasks."

gr.Interface(inference,inputs=[gr.inputs.Image(type="filepath"),gr.Slider(0.0, 1.0, value=0.5)],outputs=gr.Gallery(label="Detection Result"),title=title,description=description).launch(enable_queue=True)