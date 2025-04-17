import gradio as gr
import paddlehub as hub
import cv2

classifier = hub.Module(name="pnasnet_imagenet")

def inference(img):
  test_img_path = img
  input_dict = {"image": [test_img_path]}
  result = classifier.classification(data=input_dict)
  print(result)
  return result[0][0]

  
title="pnasnet_imagenet"
description="PNASNet is an image classification model automatically trained by Google through AutoML. The PaddleHub Module is trained based on the ImageNet-2012 dataset, accepts input images with a size of 224 x 224 x 3, and supports prediction directly through the command line or Python interface."

examples=[['cat2.jpg']]
gr.Interface(inference,gr.inputs.Image(type="filepath"),"label",title=title,description=description,examples=examples).launch(enable_queue=True,cache_examples=True)
