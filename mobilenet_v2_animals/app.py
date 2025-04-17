import gradio as gr
import cv2
import paddlehub as hub
from PIL import Image
import numpy as np

classifier = hub.Module(name="mobilenet_v2_animals")


def inference(img):
  result = classifier.classification(images=[cv2.imread(img)])
  return result[0]

  
title="mobilenet_v2_animals"
description="MobileNet V2 is a lightweight convolutional neural network. On the basis of MobileNet, it has made two major improvements: Inverted Residuals and Linear bottlenecks. The PaddleHub Module is trained on Baidu's self-built animal dataset and can be used for image classification and feature extraction. Currently, it supports the classification and recognition of 7,978 animals. Details of the model can be found in the paper ."

examples=[['rabbit.jpeg']]
gr.Interface(inference,gr.inputs.Image(type="filepath"),"label",title=title,description=description,examples=examples).launch(enable_queue=True)