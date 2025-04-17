import gradio as gr
import cv2
import paddlehub as hub
from PIL import Image
import numpy as np




model = hub.Module(name='animegan_v2_hayao_99', use_gpu=False)


def inference(img):
  result = model.style_transfer(images=[cv2.imread(img)])
  return result[0][:,:,::-1]

  
title="animegan_v2_hayao_99"
description="AnimeGAN V2 image style conversion model, the model can convert the input image into Hayao Miyazaki anime style, and the model weights are converted from the AnimeGAN V2 official open source project."

examples=[['bridge.jpg']]
gr.Interface(inference,gr.inputs.Image(type="filepath"),gr.outputs.Image(type="numpy"),title=title,description=description,examples=examples).launch(enable_queue=True)