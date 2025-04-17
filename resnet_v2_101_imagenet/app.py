import gradio as gr
import paddlehub as hub


classifier = hub.Module(name="resnet_v2_101_imagenet")
    
def inference(img):
  test_img_path = img
  input_dict = {"image": [test_img_path]}
  result = classifier.classification(data=input_dict)
  return result[0][0]

  
title="resnet_v2_101_imagenet"
description="The ResNet series model is one of the important models in the field of image classification. The residual unit proposed in the model effectively solves the difficult problem of deep network training, and improves the accuracy of the model by increasing the depth of the model. The structure of the PaddleHub Module is ResNet101, which is trained based on the ImageNet-2012 dataset. The input image size is 224 x 224 x 3, and it supports prediction directly through the command line or Python interface."

examples=[['cat2.jpg']]
gr.Interface(inference,gr.inputs.Image(type="filepath"),"label",title=title,description=description,examples=examples).launch(enable_queue=True,cache_examples=True)