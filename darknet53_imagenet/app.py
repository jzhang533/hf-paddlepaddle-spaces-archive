import gradio as gr
import paddlehub as hub


classifier = hub.Module(name="darknet53_imagenet")
    
def inference(img):
  test_img_path = img
  input_dict = {"image": [test_img_path]}
  result = classifier.classification(data=input_dict)
  return result[0][0]

  
title="darknet53_imagenet"
description="DarkNet is an image classification model proposed by Joseph Redmon and used in Yolov3 as Backbone to complete feature extraction. The network is connected with consecutive 3 3 and 1 1 convolutions and has ShortCut connections like ResNet. The PaddleHub Module is trained based on the ImageNet-2012 dataset, accepts input images with a size of 224 x 224 x 3, and supports prediction directly through the command line or Python interface."

examples=[['cat2.jpg']]
gr.Interface(inference,gr.inputs.Image(type="filepath"),"label",title=title,description=description,examples=examples).launch(enable_queue=True,cache_examples=True)