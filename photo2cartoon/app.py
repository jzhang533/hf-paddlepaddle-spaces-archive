import os
os.system("hub install Photo2Cartoon==1.0.0")
import gradio as gr
import paddlehub as hub
import cv2
from pathlib import Path

model = hub.Module(name='Photo2Cartoon')

def inference(image):
    result = model.Cartoon_GEN(
    images=[cv2.imread(image.name)],
    paths=None,
    batch_size=4,
    output_dir='output',
    visualization=True,
    use_gpu=False)
    return './output/result_0.png'
    
title = "Photo2Cartoon"
description = "Gradio demo for Photo2cartoon. To use it, simply upload your image, or click one of the examples to load them. Read more at the links below."
article = "<p style='text-align: center'><a href='https://github.com/minivision-ai/photo2cartoon' target='_blank'>Github Repo</a></p>"
iface = gr.Interface(inference, inputs=gr.inputs.Image(type="file",source="webcam"), outputs=gr.outputs.Image(type="file"),enable_queue=True,title=title,article=article,description=description)
iface.launch()