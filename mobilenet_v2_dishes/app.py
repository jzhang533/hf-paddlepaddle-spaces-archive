import gradio as gr
import paddlehub as hub
import cv2

classifier = hub.Module(name="mobilenet_v2_dishes")

def inference(img):
  result = classifier.classification(images=[cv2.imread(img)])
  print(result)
  return result[0]

  
title="mobilenet_v2_dishes"
description="MobileNet V2 is a lightweight convolutional neural network. On the basis of MobileNet, it has made two major improvements: Inverted Residuals and Linear bottlenecks. The PaddleHub Module is trained on Baidu's self-built dishes dataset and can be used for image classification and feature extraction. Currently, it supports the classification and recognition of 8416 dishes."

examples=[['dish.jpg']]
gr.Interface(inference,gr.inputs.Image(type="filepath"),"label",title=title,description=description,examples=examples).launch(enable_queue=True,cache_examples=True)