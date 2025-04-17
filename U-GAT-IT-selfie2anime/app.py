import os
import cv2
os.system("hub install UGATIT_100w==1.0.0")
import gradio as gr
import paddlehub as hub
import numpy as np
from PIL import Image

model = hub.Module(name='UGATIT_100w', use_gpu=False)

def inference(image):
    #result = model.style_transfer(images=[cv2.imread(image.name)])
    result = model.style_transfer(paths=[image.name])
    print(type(result[0]))
    print(result[0])
    return Image.fromarray(np.uint8(result[0])[:,:,::-1]).convert('RGB')
    
title = "UGATIT-selfie2anime"
description = "Gradio demo for UGATIT-selfie2anime. To use it, simply upload your image, or click one of the examples to load them. Read more at the links below."
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/1907.10830' target='_blank'>U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation</a> | <a href='https://github.com/taki0112/UGATIT' target='_blank'>Github Repo</a></p>"
examples=[['robert.png']]
iface = gr.Interface(inference, inputs=gr.inputs.Image(type="file"), outputs=gr.outputs.Image(type="pil"),examples=examples,enable_queue=True,title=title,article=article,description=description)
iface.launch()