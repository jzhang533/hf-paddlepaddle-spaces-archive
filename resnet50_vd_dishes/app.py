import gradio as gr
import paddlehub as hub
import cv2

classifier = hub.Module(name="resnet50_vd_dishes")

def inference(img):
  result = classifier.classification(images=[cv2.imread(img)])
  print(result)
  return result[0]

  
title="resnet50_vd_dishes"
description="ResNet-vd is a variant of the original structure of ResNet, which can be used for image classification and feature extraction. The PaddleHub Module is trained using Baidu's self-built dish dataset, and supports the classification and identification of 8,416 dishes."

examples=[['dish.jpg']]
gr.Interface(inference,gr.inputs.Image(type="filepath"),"label",title=title,description=description,examples=examples).launch(enable_queue=True,cache_examples=True)