import gradio as gr
import paddlehub as hub

classifier = hub.Module(name="resnet_v2_152_imagenet")

def inference(img):
  input_dict = {"image": [img]}
  result = classifier.classification(data=input_dict)
  return result[0][0]

  
title="resnet_v2_152_imagenet"
description="The ResNet series model is one of the important models in the field of image classification. The residual unit proposed in the model effectively solves the difficult problem of deep network training, and improves the accuracy of the model by increasing the depth of the model. The PaddleHub Module structure is ResNet152, based on ImageNet-2012 dataset training, accepts input images with a size of 224 x 224 x 3, and supports prediction directly through the command line or Python interface."

examples=[['rabbit.jpeg']]
gr.Interface(inference,gr.inputs.Image(type="filepath"),"label",title=title,description=description,examples=examples).launch(enable_queue=True)