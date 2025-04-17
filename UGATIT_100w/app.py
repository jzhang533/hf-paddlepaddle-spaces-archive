import gradio as gr
import cv2
import paddlehub as hub
from PIL import Image
import numpy as np

model = hub.Module(name="UGATIT_100w")


def inference(img):
  result = model.style_transfer(images=[cv2.imread(img)])
  return result[0][:,:,::-1]

  
title="UGATIT_100w"
description="UGATIT image style conversion model, the model can convert the input face image into anime style, please refer to the UGATIT-Paddle open source project for model details."

examples=[['groot.png']]
gr.Interface(inference,gr.inputs.Image(type="filepath"),gr.outputs.Image(type="numpy"),title=title,description=description,examples=examples).launch(enable_queue=True,cache_examples=True)