import gradio as gr
import paddlehub as hub
import cv2

classifier = hub.Module(name="mobilenet_v2_imagenet_ssld")

def inference(img):
  result = classifier.classification(images=[cv2.imread(img)])
  print(result)
  return result[0]

  
title="mobilenet_v2_imagenet_ssld"
description="MobileNet V2 is an image classification model proposed by Mark Sandler, Andrew Howard, etc. in 2018. This series of models (MobileNet) is an efficient model proposed for mobile and embedded devices, and it still maintains a relatively high performance with fewer model parameters. High classification accuracy. The PaddleHub Module is based on the ImageNet-2012 dataset and is trained by the SSLD distillation method provided by PaddleClas. The input image size is 224 x 224 x 3, supports finetune, and can also be predicted directly through the command line or Python interface."

examples=[['cat2.jpg']]
gr.Interface(inference,gr.inputs.Image(type="filepath"),"label",title=title,description=description,examples=examples).launch(enable_queue=True,cache_examples=True)