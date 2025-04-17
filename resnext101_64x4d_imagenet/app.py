import gradio as gr
import paddlehub as hub

classifier = hub.Module(name="resnext101_64x4d_imagenet")

def inference(img):
  test_img_path = img
  input_dict = {"image": [test_img_path]}
  result = classifier.classification(data=input_dict)
  print(result)
  return result[0][0]

  
title="resnext101_64x4d_imagenet"
description="ResNeXt is an image classification model proposed by UC San Diego and Facebook AI Research Institute in 2017. The model follows the stacking idea of ​​VGG/ResNets and adopts the split-transform-merge strategy to increase the number of branches of the network. resnext101_64x4d, indicating that the number of layers is 101, the number of branches is 64, and the input and output channels of each branch are 4. The PaddleHub Module is weakly trained on a dataset containing billions of social media images, and uses the ImageNet-2012 dataset finetune. It accepts input images with a size of 224 x 224 x 3 and supports direct command line or Python interface."

examples=[['cat2.jpg']]
gr.Interface(inference,gr.inputs.Image(type="filepath"),"label",title=title,description=description,examples=examples).launch(enable_queue=True)