import gradio as gr
import paddlehub as hub
import cv2


classifier = hub.Module(name="fix_resnext101_32x48d_wsl_imagenet")

def inference(img):
  result = classifier.classification(images=[cv2.imread(img)])
  print(result)
  return result[0]

  
title="fix_resnext101_32x48d_wsl_imagenet"
description="ResNeXt is an image classification model proposed by UC San Diego and Facebook AI Research Institute in 2017. The model follows the stacking idea of ​​VGG/ResNets and adopts the split-transform-merge strategy to increase the number of branches of the network. The PaddleHub Module is weakly trained on a dataset containing billions of social media images, and uses the ImageNet-2012 dataset finetune. It accepts input images with a size of 224 x 224 x 3 and supports direct command line or Python interface."

examples=[['cat2.jpg']]
gr.Interface(inference,gr.inputs.Image(type="filepath"),"label",title=title,description=description,examples=examples).launch(enable_queue=True,cache_examples=True)