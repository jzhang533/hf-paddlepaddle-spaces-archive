import gradio as gr
import paddlehub as hub
import cv2

classifier = hub.Module(name="efficientnetb1_imagenet")

def inference(img):
  result = classifier.classification(images=[cv2.imread(img)])
  print(result)
  return result[0]

  
title="efficientnetb1_imagenet"
description="EfficientNet is a new model of Google's open source. It is a lightweight network. Its backbone network is composed of MBConv, and the squeeze-and-excitation operation is adopted to optimize the network structure. The PaddleHub Module structure is EfficientNetB1, based on ImageNet-2012 dataset training, accepts input images with a size of 224 x 224 x 3, and supports prediction directly through the command line or Python interface."

examples=[['cat2.jpg']]
gr.Interface(inference,gr.inputs.Image(type="filepath"),"label",title=title,description=description,examples=examples).launch(enable_queue=True,cache_examples=True)