import gradio as gr
import cv2
import paddlehub as hub
from PIL import Image
import numpy as np


model = hub.Module(name='MiDaS_Small', use_gpu=False)

def inference(img):
  result = model.depth_estimation(images=[cv2.imread(img)],visualization=True)
  return './output/0.png'

  
title="MiDaS_Small"
description="MiDaS_Small is a monocular depth estimation model that estimates depth information from the input image."

examples=[['cat.png']]
gr.Interface(inference,gr.inputs.Image(type="filepath"),gr.outputs.Image(type="file"),title=title,description=description,examples=examples).launch(enable_queue=True)