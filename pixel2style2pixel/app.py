import gradio as gr
import cv2
import paddlehub as hub
from PIL import Image
import numpy as np

model = hub.Module(name='pixel2style2pixel')


def inference(content):
  res = model.style_transfer(paths=[content], output_dir='./transfer_result/', use_gpu=False)  
  print(res)
  return res[0]

  
title="pixel2style2pixel"
description="Pixel2Style2Pixel encodes images using a sizable model into the style vector space of StyleGAN V2, making the pre-encoded and decoded images strongly correlated. This module is applied to the face transformation task."

examples=[['bridgetown.jpeg']]
gr.Interface(inference,[gr.inputs.Image(type="filepath")],gr.outputs.Image(type="numpy"),title=title,description=description,examples=examples).launch(enable_queue=True)