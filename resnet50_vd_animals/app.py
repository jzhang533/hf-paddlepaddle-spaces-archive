import gradio as gr
import paddlehub as hub
import cv2

classifier = hub.Module(name="resnet50_vd_animals")

def inference(img):
  result = classifier.classification(images=[cv2.imread(img)])
  print(result)
  return result[0]

  
title="resnet50_vd_animals"
description="ResNet-vd is actually ResNet-D, a variant of the original structure of ResNet, which can be used for image classification and feature extraction. The PaddleHub Module is trained using Baidu's self-built animal dataset, and supports the classification and recognition of 7978 animals."

examples=[['cat2.jpg']]
gr.Interface(inference,gr.inputs.Image(type="filepath"),"label",title=title,description=description,examples=examples).launch(enable_queue=True)