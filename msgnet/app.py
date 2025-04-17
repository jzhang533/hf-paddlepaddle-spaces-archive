import gradio as gr
import cv2
import paddlehub as hub
from PIL import Image
import numpy as np

model = hub.Module(name='msgnet')


def inference(img,style):
  res = model.predict(origin=[img], style=style, visualization=False)
  return res[0][:,:,::-1]

  
title="msgnet"
description="Multi-style Generative Network for Real-time Transfer"

examples=[['bridgetown.jpeg','starry.jpeg']]
gr.Interface(inference,[gr.inputs.Image(type="filepath"),gr.inputs.Image(type="filepath")],gr.outputs.Image(type="numpy"),title=title,description=description,examples=examples).launch(enable_queue=True)