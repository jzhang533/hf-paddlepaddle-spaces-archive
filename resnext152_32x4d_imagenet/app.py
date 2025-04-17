import gradio as gr
import cv2
import paddlehub as hub
from PIL import Image
import numpy as np

classifier = hub.Module(name="resnext152_32x4d_imagenet")

output = {}
def inference(img):
  input_dict = {"image": [img]}
  result = classifier.classification(data=input_dict)
  return result[0][0]

  
title="resnext152_32x4d_imagenet"
description="ResNeXt is an image classification model proposed by UC San Diego and Facebook AI Research Institute in 2017. The model follows the stacking idea of ​​VGG/ResNets and adopts the split-transform-merge strategy to increase the number of branches of the network. resnext152_32x4d, indicating that the number of layers is 152, the number of branches is 32, and the input and output channels of each branch are 4. The PaddleHub Module is weakly trained on a dataset containing billions of social media images, and uses the ImageNet-2012 dataset finetune. It accepts input images with a size of 224 x 224 x 3 and supports direct command line or Python interface. predict."

examples=[['rabbit.jpeg']]
gr.Interface(inference,gr.inputs.Image(type="filepath"),"label",title=title,description=description,examples=examples).launch(enable_queue=True)