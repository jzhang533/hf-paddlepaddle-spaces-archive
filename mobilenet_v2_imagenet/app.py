import gradio as gr
import paddlehub as hub


classifier = hub.Module(name="mobilenet_v2_imagenet")
    
def inference(img):
  test_img_path = img
  input_dict = {"image": [test_img_path]}
  result = classifier.classification(data=input_dict)
  return result[0][0]

  
title="mobilenet_v2_imagenet"
description="MobileNet V2 is an image classification model proposed by Mark Sandler, Andrew Howard, etc. in 2018. This series of models (MobileNet) is an efficient model proposed for mobile and embedded devices, and it still maintains a relatively high performance with fewer model parameters. High classification accuracy. The PaddleHub Module is trained based on the ImageNet-2012 dataset, accepts input images with a size of 224 x 224 x 3, and supports prediction directly through the command line or Python interface."

examples=[['cat2.jpg']]
gr.Interface(inference,gr.inputs.Image(type="filepath"),"label",title=title,description=description,examples=examples).launch(enable_queue=True,cache_examples=True)