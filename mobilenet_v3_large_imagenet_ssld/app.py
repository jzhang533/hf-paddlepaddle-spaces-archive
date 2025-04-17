import gradio as gr
import paddlehub as hub
import cv2

classifier = hub.Module(name="mobilenet_v3_large_imagenet_ssld")

def inference(img):
  result = classifier.classification(images=[cv2.imread(img)])
  return result[0]

  
title="mobilenet_v3_large_imagenet_ssld"
description="MobileNetV3 is a new model released by Google in 2019. The author obtains the network structure by combining NAS and NetAdapt to search, and provides two versions, Large and Small, which are suitable for different resource requirements. Compared with MobileNetV2, the new model has improved speed and accuracy. The model structure of the PaddleHubModule is MobileNetV3 Large, which is based on the ImageNet-2012 dataset and is trained by the SSLD distillation method provided by PaddleClas. The input image size is 224 x 224 x 3, and it supports finetune. It can also be performed directly through the command line or Python interface. predict."

examples=[['cat2.jpg']]
gr.Interface(inference,gr.inputs.Image(type="filepath"),"label",title=title,description=description,examples=examples).launch(enable_queue=True)