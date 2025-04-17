import gradio as gr
import paddlehub as hub
import cv2


classifier = hub.Module(name="vgg13_imagenet")

def inference(img):
  test_img_path = img
  input_dict = {"image": [test_img_path]}
  result = classifier.classification(data=input_dict)
  print(result)
  return result[0][0]

  
title="vgg13_imagenet"
description="VGG is an image classification model proposed by the Oxford University Computer Vision Group and DeepMind in 2014. This series of models explores the relationship between the depth of the convolutional neural network and its performance. It is experimentally proved that increasing the depth of the network can affect the final performance of the network to a certain extent. So far, VGG is still used by many other image tasks. BackBone network for feature extraction. The PaddleHub Module has a VGG13 structure and is trained on the ImageNet-2012 dataset. The input image size is 224 x 224 x 3, and it supports prediction directly through the command line or Python interface."

examples=[['cat2.jpg']]
gr.Interface(inference,gr.inputs.Image(type="filepath"),"label",title=title,description=description,examples=examples).launch(enable_queue=True,cache_examples=True)