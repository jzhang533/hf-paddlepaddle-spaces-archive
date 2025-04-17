import gradio as gr
import paddlehub as hub


classifier = hub.Module(name="dpn107_imagenet")
    
def inference(img):
  test_img_path = img
  input_dict = {"image": [test_img_path]}
  result = classifier.classification(data=input_dict)
  return result[0][0]

  
title="dpn107_imagenet"
description="DPN (Dual Path Networks) is the image classification model of the ImageNet 2017 target positioning champion, which combines the core ideas of ResNet and DenseNet. The PaddleHub Module structure is DPN107, based on ImageNet-2012 dataset training, accepts input images with a size of 224 x 224 x 3, and supports prediction directly through the command line or Python interface."

examples=[['cat2.jpg']]
gr.Interface(inference,gr.inputs.Image(type="filepath"),"label",title=title,description=description,examples=examples).launch(enable_queue=True,cache_examples=True)