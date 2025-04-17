import gradio as gr
import cv2
import paddlehub as hub
from PIL import Image
import numpy as np

model = hub.Module(name='repvgg_a0_imagenet')

output = {}
def inference(img):
  result = model.predict([img])
  for key, value in result[0].items():
    output[key] = float(value)
  return output

  
title="repvgg_a0_imagenet"
description="The RepVGG (Making VGG-style ConvNets Great Again) series model is a simple but functional model proposed by Tsinghua University (Ding Guiguang's team), Megvii Technology (Sun Jian, etc.), Hong Kong University of Science and Technology and Aberystwyth University in 2021. Powerful convolutional neural network architecture. There is an inference time agent similar to VGG. The body consists of 3x3 convolutions and relu stacks, while the training-time model has a multi-branch topology. The decoupling of training time and inference time is achieved through a reparameterization technique, so the model is called repvgg."

examples=[['rabbit.jpeg']]
gr.Interface(inference,gr.inputs.Image(type="filepath"),"label",title=title,description=description,examples=examples).launch(enable_queue=True)