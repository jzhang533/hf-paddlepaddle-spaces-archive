import gradio as gr
import cv2
import paddlehub as hub
from PIL import Image
import numpy as np

model = hub.Module(name='stylepro_artistic')


def inference(content,style):
  res = model.style_transfer(images=[{'content': cv2.imread(content),'styles': [cv2.imread(style)]}])
  return res[0]['data'][:,:,::-1]

  
title="stylepro_artistic"
description="Art style transfer models can convert a given image to an arbitrary art style. This model, StyleProNet, adopts a fully convolutional neural network architecture (FCNs) as a whole, and reconstructs artistic style pictures through encoder-decoder. The core of StyleProNet is the unparameterized content-style fusion algorithm Style Projection, which has a small model size and fast response speed. The loss function of model training includes style loss, content perceptual loss and content KL loss, which ensures that the model can restore the semantic details of content pictures and style information of style pictures with high fidelity. The pre-training dataset uses the MS-COCO dataset as the content-side images, and the WikiArt dataset as the style-side images. For more details, please refer to the StyleProNet paper https://arxiv.org/abs/2003.07694 ."

examples=[['bridgetown.jpeg','starry.jpeg']]
gr.Interface(inference,[gr.inputs.Image(type="filepath"),gr.inputs.Image(type="filepath")],gr.outputs.Image(type="numpy"),title=title,description=description,examples=examples).launch(enable_queue=True)