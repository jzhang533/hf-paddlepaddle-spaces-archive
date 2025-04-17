import gradio as gr
import cv2
import paddlehub as hub
from PIL import Image
import numpy as np


model = hub.Module(name='MiDaS_Large', use_gpu=False)

def inference(img):
  model.depth_estimation(images=[cv2.imread(img)],visualization=True)
  return './output/0.png'

  
title="MiDaS_Large"
description="MiDaS_Large is a monocular depth estimation model that estimates depth information from input images."

examples=[['lion.jpg']]
gr.Interface(inference,gr.inputs.Image(type="filepath"),gr.outputs.Image(type="file"),title=title,description=description,examples=examples).launch(enable_queue=True,debug=True)